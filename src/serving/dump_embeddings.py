import torch
import logging
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.rgcn import HeteroRGCN
from src.data.pyg_graph_conversion import load_hetero_data
from src.serving.storage import store_embedding
from src.serving.config import MODEL_PATH, EMBEDDING_DIM
import torch_geometric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def strip_model_prefix(state_dict):
    # Remove 'model.' prefix from all keys if present
    return {k.replace('model.', '', 1) if k.startswith('model.') else k: v for k, v in state_dict.items()}

def dump_embeddings():
    """Dump all node embeddings from the trained model to Redis."""
    try:
        logger.info("Loading model and data...")
        data = load_hetero_data()
        
        # Print feature dimensions for each node type
        logger.info("Current feature dimensions:")
        for node_type in data.node_types:
            feat = data[node_type].x
            logger.info(f"{node_type} feature shape: {feat.shape}")
        
        # Get input dimensions from the first layer of the model
        state_dict = torch.load(MODEL_PATH)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = strip_model_prefix(state_dict)
        
        # Print expected input dimensions from model weights
        logger.info("\nModel's expected input dimensions:")
        for key in state_dict.keys():
            if 'conv1.convs' in key and 'weight' in key:
                weight = state_dict[key]
                logger.info(f"{key}: {weight.shape}")
        
        # Load and prepare model
        model = HeteroRGCN(
            metadata=data.metadata(),
            hidden_channels=EMBEDDING_DIM
        )
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH)
        if isinstance(checkpoint, dict) and all(k.startswith('model.') for k in checkpoint.keys()):
            checkpoint = {k.replace('model.', '', 1): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
        model.eval()

        # Diagnostic printouts
        logger.info("Expected input dimensions from checkpoint:")
        for key in checkpoint.keys():
            if 'conv1.convs' in key and 'weight' in key:
                logger.info(f"{key}: {checkpoint[key].shape}")
        logger.info("\nActual input feature shapes from data:")
        for node_type in ['user', 'post', 'subreddit']:
            logger.info(f"{node_type} feature shape: {data[node_type].x.shape}")

        # Print edge_type tensors for each edge type before model forward
        logger.info("\nEdge type tensors before model forward:")
        for edge_type in data.edge_types:
            et = data[edge_type].edge_type if hasattr(data[edge_type], 'edge_type') else None
            if et is not None:
                logger.info(f"{edge_type}: shape={et.shape}, first 10={et[:10].tolist()}, unique={torch.unique(et).tolist()}")
            else:
                logger.info(f"{edge_type}: No edge_type tensor found.")

        # Print EDGE_TYPE_TO_ID mapping if available
        try:
            from src.data.pyg_graph_conversion import EDGE_TYPE_TO_ID
            logger.info(f"\nEDGE_TYPE_TO_ID mapping: {EDGE_TYPE_TO_ID}")
        except Exception as e:
            logger.info(f"\nCould not import EDGE_TYPE_TO_ID: {e}")

        # Print the order of edge types in metadata
        logger.info(f"\nOrder of edge types in data.metadata()[1]:\n{data.metadata()[1]}")

        # Print the order of edge types as expected by the model (from checkpoint keys)
        model_edge_types = []
        for key in checkpoint.keys():
            if key.startswith('conv1.convs.<') and key.endswith('.weight'):
                etype = key[len('conv1.convs.<'):-len('>.weight')]
                etype_tuple = tuple(etype.split('___'))
                model_edge_types.append(etype_tuple)
        logger.info(f"\nOrder of edge types as expected by the model (from checkpoint):\n{model_edge_types}")

        # Print model's internal mapping from edge type ID to edge type tuple
        try:
            conv1 = model.conv1.convs
            logger.info("\nModel's internal mapping (conv1):")
            for i, (etype, conv) in enumerate(conv1.items()):
                logger.info(f"ID {i}: {etype}")
        except Exception as e:
            logger.info(f"\nCould not print model's internal mapping: {e}")

        # Print actual shapes of root weights for each edge type in the loaded model
        logger.info("\nActual shapes of root weights in the loaded model:")
        for etype, conv in model.conv1.convs.items():
            if hasattr(conv, 'root'):
                root = conv.root
                if root is not None:
                    logger.info(f"{etype}: root weight shape={root.shape}")
                else:
                    logger.info(f"{etype}: No root weight (root_weight=False)")

        # Print model's internal logic for selecting the root weight for each edge type
        logger.info("\nModel's internal logic for selecting the root weight for each edge type:")
        for etype, conv in model.conv1.convs.items():
            if hasattr(conv, 'root'):
                root = conv.root
                if root is not None:
                    logger.info(f"{etype}: root weight shape={root.shape}, root weight type={root.dtype}, root weight device={root.device}")
                else:
                    logger.info(f"{etype}: No root weight (root_weight=False)")

        # Print torch-geometric version
        logger.info(f"\ntorch-geometric version: {torch_geometric.__version__}")

        # Print dtype, min, max, unique values of all edge_type tensors
        logger.info("\nEdge type tensor stats:")
        for edge_type in data.edge_types:
            et = data[edge_type].edge_type if hasattr(data[edge_type], 'edge_type') else None
            if et is not None:
                logger.info(f"{edge_type}: dtype={et.dtype}, min={et.min().item()}, max={et.max().item()}, unique={torch.unique(et).tolist()}")
            else:
                logger.info(f"{edge_type}: No edge_type tensor found.")

        logger.info("Computing embeddings...")
        with torch.no_grad():
            # Prepare input dictionaries
            x_dict = {ntype: data[ntype].x for ntype in data.node_types}
            edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}
            edge_type_dict = {etype: data[etype].edge_type for etype in data.edge_types if hasattr(data[etype], 'edge_type')}

            # Print shape, dtype, and first row of x_dict['post']
            post_x = x_dict['post']
            logger.info(f"\nx_dict['post']: shape={post_x.shape}, dtype={post_x.dtype}, first row={post_x[0].tolist() if post_x.shape[0] > 0 else 'EMPTY'}")

            # Minimal forward pass on just the ('post', 'self', 'post') edge type
            try:
                logger.info("\nAttempting minimal forward pass on ('post', 'self', 'post') edge type...")
                etype = ('post', 'self', 'post')
                conv = model.conv1.convs[etype]
                edge_index = edge_index_dict[etype]
                edge_type = edge_type_dict[etype]
                out = conv(x_dict['post'], edge_index, edge_type)
                logger.info(f"Minimal forward pass output shape: {out.shape}")
            except Exception as e:
                logger.info(f"Minimal forward pass error: {e}")

            # For each edge type, print source/target node type, input feature shape, and unique edge_type values
            logger.info("\nPer-edge-type diagnostics:")
            for edge_type in data.edge_types:
                src, rel, dst = edge_type
                src_feat = x_dict[src]
                et = edge_type_dict.get(edge_type, None)
                logger.info(f"Edge type: {edge_type}")
                logger.info(f"  Source node type: {src}, feature shape: {src_feat.shape}")
                logger.info(f"  Target node type: {dst}")
                if et is not None:
                    logger.info(f"  edge_type tensor: shape={et.shape}, unique={torch.unique(et).tolist()}")
                else:
                    logger.info(f"  edge_type tensor: None")

            # Compute embeddings
            embeddings = model(x_dict, edge_index_dict, edge_type_dict)
        
        logger.info("Storing embeddings in Redis...")
        for node_type in data.node_types:
            if embeddings[node_type] is not None:
                node_embeddings = embeddings[node_type]
                num_nodes = node_embeddings.size(0)
                logger.info(f"Storing {num_nodes} {node_type} embeddings...")
                
                for node_id in range(num_nodes):
                    embedding = node_embeddings[node_id].cpu().numpy()
                    store_embedding(node_type, node_id, embedding)
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error during embedding dump: {str(e)}")
        raise

if __name__ == "__main__":
    dump_embeddings() 