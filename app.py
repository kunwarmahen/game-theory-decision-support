from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import uvicorn
from typing import Optional, Dict, List, Any
import requests
import logging
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates for the frontend
templates = Jinja2Templates(directory="templates")

# Pydantic models for request validation
class AnalysisRequest(BaseModel):
    query: str
    ollama_url: str = settings.default_ollama_url
    model_name: str = settings.default_model_name


def normalize_probabilities(nodes: List[Dict], parent_children_map: Dict[str, List[str]]) -> List[Dict]:
    """
    Normalize probabilities so that children of each parent sum to 100%.

    Args:
        nodes: List of decision tree nodes
        parent_children_map: Dictionary mapping parent node IDs to their children IDs

    Returns:
        List of nodes with normalized probabilities
    """
    logger.info("Normalizing decision tree probabilities")

    # Create a lookup for nodes by ID
    node_lookup = {node['id']: node for node in nodes}

    # For each parent, normalize its children's probabilities
    for parent_id, children_ids in parent_children_map.items():
        # Get children nodes that have probabilities
        children_with_probs = []
        for child_id in children_ids:
            if child_id in node_lookup and 'probability' in node_lookup[child_id]:
                children_with_probs.append(node_lookup[child_id])

        if not children_with_probs:
            continue

        # Calculate total probability
        total_prob = sum(child.get('probability', 0) for child in children_with_probs)

        # Normalize if total is not 0
        if total_prob > 0:
            for child in children_with_probs:
                child['probability'] = child.get('probability', 0) / total_prob
                logger.debug(f"Normalized {child['id']} probability to {child['probability']:.2%}")
        else:
            # If all probabilities are 0, distribute equally
            equal_prob = 1.0 / len(children_with_probs)
            for child in children_with_probs:
                child['probability'] = equal_prob
                logger.debug(f"Set {child['id']} to equal probability {equal_prob:.2%}")

    return nodes


def validate_and_fix_decision_tree(tree_data: List[Dict]) -> List[Dict]:
    """
    Validate and fix the decision tree structure to ensure:
    1. All nodes have required fields
    2. Probabilities at each level sum to 100%
    3. Tree structure is valid

    Args:
        tree_data: List of decision tree nodes

    Returns:
        Validated and fixed decision tree
    """
    logger.info("Validating and fixing decision tree structure")

    if not tree_data:
        logger.warning("Empty decision tree provided")
        return tree_data

    # Build parent-children relationships
    parent_children_map = {}
    for node in tree_data:
        if 'children' in node and node['children']:
            parent_children_map[node['id']] = node['children']

    # Normalize probabilities
    tree_data = normalize_probabilities(tree_data, parent_children_map)

    # Ensure all nodes have required fields
    for node in tree_data:
        if 'type' not in node:
            node['type'] = 'outcome'
        if 'label' not in node:
            node['label'] = node.get('id', 'Unknown')
        if 'children' not in node:
            node['children'] = []

    logger.info(f"Decision tree validated with {len(tree_data)} nodes")
    return tree_data


def normalize_outcomes(outcomes: List[Dict]) -> List[Dict]:
    """
    Normalize outcome probabilities to sum to 100%.

    Args:
        outcomes: List of outcome dictionaries

    Returns:
        List of outcomes with normalized probabilities
    """
    logger.info("Normalizing outcome probabilities")

    if not outcomes:
        return outcomes

    # Calculate total probability
    total_prob = sum(outcome.get('probability', 0) for outcome in outcomes)

    # Normalize if total is not 0
    if total_prob > 0:
        for outcome in outcomes:
            outcome['probability'] = outcome.get('probability', 0) / total_prob
            logger.debug(f"Normalized outcome '{outcome.get('description', 'Unknown')}' to {outcome['probability']:.2%}")
    else:
        # If all probabilities are 0, distribute equally
        equal_prob = 1.0 / len(outcomes)
        for outcome in outcomes:
            outcome['probability'] = equal_prob
            logger.debug(f"Set outcome to equal probability {equal_prob:.2%}")

    return outcomes


class GameTheoryAnalyzer:
    def __init__(self, ollama_url, model_name):
        self.ollama_url = ollama_url
        self.model_name = model_name
        
    def analyze_situation(self, user_query):
        """
        Use Ollama to analyze the situation and generate possible outcomes with probabilities
        """
        prompt = f"""
        You are an expert in game theory and decision analysis using principles from Nash Equilibrium,
        Expected Value Theory, and Strategic Decision Making.

        Analyze this situation: {user_query}

        CRITICAL REQUIREMENTS FOR PROBABILITIES:
        1. For the "outcomes" array: probabilities must sum to EXACTLY 1.0
        2. For the "decision_tree": at each branch point, the probabilities of all children must sum to EXACTLY 1.0
        3. Every node with a "probability" field represents a chance event with that likelihood
        4. Probabilities should be realistic and based on game theory principles

        DECISION TREE STRUCTURE:
        - "decision" nodes: Represent choices the user can make (no probability field)
        - "chance" nodes: Represent uncertain events (must have probability field)
        - "outcome" nodes: Represent final results (should have probability if child of chance node)
        - The root node should typically be a "decision" node
        - For each parent node, list ALL its children, and their probabilities must sum to 1.0

        Return your analysis as a JSON object with this EXACT structure:
        {{
            "stakeholders": ["stakeholder1", "stakeholder2", ...],
            "summary": "A clear, concise summary of the strategic situation",
            "outcomes": [
                {{
                    "description": "Clear description of this outcome",
                    "probability": 0.XX,
                    "key_factors": ["factor1", "factor2", "factor3"],
                    "recommendation": "Specific actionable advice for achieving this outcome"
                }}
            ],
            "recommended_outcome": "Description of the most strategically favorable outcome based on expected value",
            "decision_tree": [
                {{
                    "id": "root",
                    "label": "Initial Decision",
                    "children": ["option1", "option2"],
                    "type": "decision"
                }},
                {{
                    "id": "option1",
                    "label": "Choose Option 1",
                    "probability": 0.60,
                    "children": ["outcome1", "outcome2"],
                    "type": "chance"
                }},
                {{
                    "id": "outcome1",
                    "label": "Successful Result",
                    "probability": 0.40,
                    "children": [],
                    "type": "outcome"
                }}
            ]
        }}

        IMPORTANT VALIDATION RULES:
        - Provide 3-5 distinct outcomes in the outcomes array
        - Decision tree should have 5-12 nodes for clarity
        - Each outcome should have 2-4 key factors
        - Use realistic probability estimates based on game theory
        - Probabilities at each level MUST sum to 1.0
        - Format response as VALID JSON ONLY - no markdown, no explanations, just the JSON object
        """
        
        try:
            # Prepare the request payload for Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 2000
                }
            }
            
            # Make the request to Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ollama API returned error: {response.text}"
                )
            
            # Parse the response
            response_data = response.json()
            response_text = response_data.get("response", "")
            
            # Try to extract JSON from the response
            # Find the first occurrence of '{' and the last occurrence of '}'
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Could not extract valid JSON from the LLM response"
                )
            
            json_text = response_text[start_idx:end_idx]
            
            # Parse the JSON response
            try:
                analysis = json.loads(json_text)

                # Validate the required fields
                required_fields = ["stakeholders", "summary", "outcomes", "recommended_outcome", "decision_tree"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = [] if field in ["stakeholders", "outcomes", "decision_tree"] else "Not provided"

                # Ensure each outcome has all required fields
                for outcome in analysis.get("outcomes", []):
                    if "probability" not in outcome:
                        outcome["probability"] = 0.0
                    if "key_factors" not in outcome:
                        outcome["key_factors"] = []
                    if "recommendation" not in outcome:
                        outcome["recommendation"] = "Not provided"
                    if "description" not in outcome:
                        outcome["description"] = "Unnamed outcome"

                # Normalize outcomes probabilities to sum to 100%
                if analysis.get("outcomes"):
                    analysis["outcomes"] = normalize_outcomes(analysis["outcomes"])

                # Validate and normalize decision tree probabilities
                if analysis.get("decision_tree"):
                    analysis["decision_tree"] = validate_and_fix_decision_tree(analysis["decision_tree"])

                logger.info("Successfully analyzed situation and normalized probabilities")
                return analysis
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid JSON in LLM response: {str(e)}\nResponse: {json_text}"
                )
        except requests.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error connecting to Ollama: {str(e)}"
            )
    
    def draw_decision_tree(self, tree_data):
        """
        Create a visualization of the decision tree and return as base64 encoded image
        """
        G = nx.DiGraph()
        
        # Process nodes
        node_types = {}
        for node in tree_data:
            G.add_node(node['id'], label=node['label'])
            node_types[node['id']] = node['type']
            
            # Add edges
            if 'children' in node:
                for child in node['children']:
                    prob = ""
                    for n in tree_data:
                        if n['id'] == child and 'probability' in n:
                            prob = f" ({n['probability']:.0%})"
                    G.add_edge(node['id'], child, label=prob)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Define node colors based on type
        node_colors = []
        for node in G.nodes():
            if node_types[node] == 'decision':
                node_colors.append('lightblue')
            elif node_types[node] == 'chance':
                node_colors.append('lightgreen')
            else:  # outcome
                node_colors.append('lightsalmon')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=20)
        
        # Add labels
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Add edge labels (probabilities)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Decision', markerfacecolor='lightblue', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Chance', markerfacecolor='lightgreen', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Outcome', markerfacecolor='lightsalmon', markersize=15)
        ]
        plt.legend(handles=legend_elements)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Convert plot to base64 encoded image
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64

# Ensure templates directory exists (template file should be manually maintained)
os.makedirs("templates", exist_ok=True)

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_situation(request: AnalysisRequest):
    """
    Analyze a strategic situation using game theory principles.

    Args:
        request: AnalysisRequest containing query, ollama_url, and model_name

    Returns:
        JSON object with analysis results including outcomes and decision tree
    """
    logger.info(f"Received analysis request for query: {request.query[:100]}...")
    logger.info(f"Using model: {request.model_name} at {request.ollama_url}")

    try:
        # Initialize analyzer with provided Ollama URL and model name
        analyzer = GameTheoryAnalyzer(request.ollama_url, request.model_name)

        # Get analysis results
        analysis = analyzer.analyze_situation(request.query)

        # Generate decision tree visualization
        if "decision_tree" in analysis and analysis["decision_tree"]:
            logger.info("Generating decision tree visualization")
            try:
                tree_image = analyzer.draw_decision_tree(analysis["decision_tree"])
                analysis["decision_tree_image"] = tree_image
            except Exception as viz_error:
                logger.error(f"Failed to generate decision tree visualization: {str(viz_error)}")
                # Continue without visualization rather than failing the entire request
                analysis["decision_tree_image"] = None
        else:
            logger.warning("No decision tree data provided in analysis")
            analysis["decision_tree_image"] = None

        logger.info("Analysis completed successfully")
        return analysis

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except requests.RequestException as e:
        logger.error(f"Network error connecting to Ollama: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to Ollama service at {request.ollama_url}. Please ensure Ollama is running and accessible."
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse LLM response. The model may not have generated valid JSON. Try again or use a different model."
        )
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# Main function to run the server
def main():
    """
    Start the FastAPI server with configuration from settings.

    Can be run directly or via uvicorn command line.
    """
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Server configuration: {settings.app_host}:{settings.app_port}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Default Ollama URL: {settings.default_ollama_url}")
    logger.info(f"Default model: {settings.default_model_name}")

    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_level=settings.log_level.lower()
    )

if __name__ == "__main__":
    main()