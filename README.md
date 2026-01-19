# Game Theory Decision Analyzer

A professional web application that uses advanced AI models and game theory principles to analyze strategic situations and provide data-driven decision support.

## Features

- **AI-Powered Analysis**: Leverages local LLM models via Ollama for intelligent strategic analysis
- **Game Theory Principles**: Applies Nash Equilibrium, Expected Value Theory, and Strategic Decision Making
- **Interactive Decision Trees**: Visual representation of decision paths and outcomes
- **Probability Analysis**: Automatic normalization ensuring probabilities sum to 100% at each level
- **Professional UI**: Modern, responsive design built with Tailwind CSS
- **Production-Ready**: Complete with logging, error handling, and deployment configurations

## Screenshots

The application features:
- Clean, modern interface with gradient headers
- Interactive decision tree visualizations
- Color-coded probability indicators
- Stakeholder and recommendation analysis
- Detailed outcome breakdowns with key factors

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **AI/LLM**: Ollama (supports Llama 3, Mistral, Mixtral, Gemma, Qwen, and more)
- **Visualization**: Matplotlib, NetworkX, Graphviz
- **Deployment**: systemd service, Nginx reverse proxy (optional)

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running
- graphviz system package

### Local Development

1. **Clone or navigate to the repository**

```bash
cd /home/mahen/Documents/ai/game_theory/decision_support
```

2. **Install system dependencies** (Ubuntu/Debian)

```bash
sudo apt install -y graphviz libgraphviz-dev pkg-config
```

3. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

5. **Configure environment** (optional)

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

6. **Install and start Ollama**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3
```

7. **Run the application**

```bash
python app.py
```

8. **Access the application**

Open your browser and navigate to: `http://localhost:8000`

## Usage

1. **Describe Your Situation**: Enter a strategic situation or dilemma in the text area
   - Example: "I'm negotiating a salary for a new job offer. The company offered $80,000 but market rate is $95,000. How should I approach this?"

2. **Select Model**: Choose your preferred LLM model from the dropdown
   - Default: Llama 3
   - Supports: Mistral, Mixtral, Gemma, Qwen, custom models

3. **Configure Settings** (Optional):
   - Click "Advanced Settings" to modify Ollama URL
   - Default: http://localhost:11434

4. **Analyze**: Click the "Analyze Situation" button

5. **Review Results**:
   - Situation summary and stakeholder analysis
   - Recommended approach
   - Possible outcomes with probabilities (automatically normalized to 100%)
   - Interactive decision tree visualization

## Configuration

Configuration is managed through environment variables (`.env` file):

```ini
# Application Settings
APP_NAME="Game Theory Decision Analyzer"
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=false
DEBUG=false

# Ollama Default Settings
DEFAULT_OLLAMA_URL=http://localhost:11434
DEFAULT_MODEL_NAME=llama3

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log

# CORS Settings
CORS_ORIGINS=*
```

## Production Deployment

For production deployment on a traditional Linux server, see [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive instructions including:

- systemd service configuration
- Nginx reverse proxy setup
- SSL/HTTPS configuration
- Firewall configuration
- Monitoring and logging
- Backup procedures

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

### Main Endpoint

**POST** `/api/analyze`

Request body:
```json
{
  "query": "Your strategic situation description",
  "ollama_url": "http://localhost:11434",
  "model_name": "llama3"
}
```

Response:
```json
{
  "stakeholders": ["stakeholder1", "stakeholder2"],
  "summary": "Analysis summary",
  "outcomes": [
    {
      "description": "Outcome description",
      "probability": 0.35,
      "key_factors": ["factor1", "factor2"],
      "recommendation": "Recommended action"
    }
  ],
  "recommended_outcome": "Most favorable outcome",
  "decision_tree": [...],
  "decision_tree_image": "base64_encoded_image"
}
```

## Key Features Explained

### Probability Normalization

The application automatically normalizes probabilities to ensure:
- All outcome probabilities sum to 100%
- At each decision tree level, child node probabilities sum to 100%
- Handles cases where the LLM provides incorrect probability distributions

### Decision Tree Types

- **Decision Nodes** (Blue): User-controlled choices
- **Chance Nodes** (Green): Uncertain events with probabilities
- **Outcome Nodes** (Salmon): Final results

### Error Handling

Production-ready error handling includes:
- Network connection errors (Ollama unavailable)
- JSON parsing errors (invalid LLM responses)
- Visualization failures (graceful degradation)
- Comprehensive logging for debugging

## Development

### Project Structure

```
decision_support/
├── app.py                          # Main application file
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment variables
├── templates/
│   └── index.html                  # Frontend template
├── game-theory-analyzer.service    # systemd service file
├── DEPLOYMENT.md                   # Deployment guide
└── README.md                       # This file
```

### Adding New Features

The application is modular and extensible:
- Add new LLM providers by extending `GameTheoryAnalyzer`
- Customize the frontend in `templates/index.html`
- Add new endpoints in `app.py`
- Configure new settings in `config.py`

## Troubleshooting

### Ollama Connection Error

```bash
# Check if Ollama is running
sudo systemctl status ollama

# Or start it manually
ollama serve
```

### Graphviz/pygraphviz Issues

```bash
# Reinstall system dependencies
sudo apt install -y graphviz libgraphviz-dev pkg-config

# Reinstall Python package
pip uninstall pygraphviz
pip install pygraphviz --no-cache-dir
```

### Port Already in Use

```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill the process or change APP_PORT in .env
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional visualization types
- More LLM provider integrations
- Enhanced UI/UX features
- Additional game theory algorithms
- Performance optimizations

## License

This project is provided as-is for educational and business use.

## Support

For issues, questions, or feature requests:
1. Check the logs in `app.log`
2. Review the [DEPLOYMENT.md](DEPLOYMENT.md) guide
3. Consult Ollama documentation: https://ollama.com/docs
4. Review FastAPI docs: https://fastapi.tiangolo.com/

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- Powered by [Ollama](https://ollama.com/)
- Visualization with [NetworkX](https://networkx.org/) and [Matplotlib](https://matplotlib.org/)
