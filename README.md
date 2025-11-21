# 🔍 Hallucination Detector

A Streamlit web application that leverages the `uqlm` (Uncertainty Quantification for Language Models) library to detect and quantify hallucinations in large language model outputs. This tool helps evaluate the reliability of AI-generated content by providing confidence scores based on multiple detection methods.

## 📋 Features

- **Multiple Detection Methods**:
  - **Black-Box Scorer**: Generates multiple responses for a prompt and compares their consistency using semantic similarity
  - **White-Box Scorer**: Analyzes token-by-token probabilities to determine model confidence
  - **LLM-as-a-Judge**: Uses multiple LLMs to evaluate whether outputs contain hallucinations
  - **Ensemble Scorer**: Combines multiple methods for more robust evaluation

- **Model Selection**: Compatible with Google's Gemini models (1.0, 1.5, 2.0)

- **Visualization**: Plotly charts showing confidence scores with interpretations

- **Customization**:
  - Adjustable temperature settings
  - Configurable scorer parameters
  - Example prompts for easy testing

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ 
- Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hallucination-detector.git
   cd hallucination-detector
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Create a `.env` file with your API key:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```
   
   Or you can enter it directly in the web interface.

### Running the App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 💡 Usage

1. Enter your Gemini API Key in the sidebar
2. Select a model and hallucination detection method
3. Configure method-specific parameters if needed
4. Enter your prompt or select an example
5. Click "Generate and Score" 
6. View the LLM response and confidence scores

### Interpreting Scores

- **Higher scores (closer to 1.0)** indicate higher confidence and lower likelihood of hallucination
- **Lower scores (closer to 0.0)** suggest potential hallucination

## 🔧 How It Works

### Black-Box Scoring

Generates multiple responses to the same prompt and measures their consistency using semantic similarity metrics. Consistent responses across multiple generations suggest higher factuality.

```python
from uqlm import BlackBoxUQ
bbuq = BlackBoxUQ(llm=llm, scorers=["semantic_negentropy"], use_best=True)
results = await bbuq.generate_and_score(prompts=[prompt], num_responses=5)
```

### White-Box Scoring

Analyzes the model's internal token probabilities to estimate uncertainty. Lower token probabilities may indicate the model is uncertain about parts of its output.

```python
from uqlm import WhiteBoxUQ
wbuq = WhiteBoxUQ(llm=llm, scorers=["min_probability"])
results = await wbuq.generate_and_score(prompts=[prompt])
```

### LLM-as-a-Judge

Uses other LLMs to evaluate the primary model's output for factual accuracy and hallucination.

```python
from uqlm import LLMPanel
llm_panel = LLMPanel(llm=llm, judges=[judge_llm1, judge_llm2])
results = await llm_panel.generate_and_score(prompts=[prompt])
```

### Ensemble Scoring

Combines multiple scoring methods for more robust evaluation.

```python
from uqlm import UQEnsemble
ensemble = UQEnsemble(llm=llm, scorers=["exact_match", "min_probability", llm])
results = await ensemble.generate_and_score(prompts=[prompt])
```

## 📚 Best Practices

- Use multiple detection methods for more reliable hallucination detection
- Lower temperature settings typically produce fewer hallucinations
- Compare model responses to known factual information when possible
- For ensemble scoring, providing ground truth can improve calibration

## ⚖️ Limitations

- Detection methods are probabilistic and not guaranteed to catch all hallucinations
- Different models may require different threshold interpretations
- Performance may vary based on prompt complexity and domain

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [uqlm library](https://github.com/uq-lm/uqlm) for hallucination detection methods
- [Streamlit](https://streamlit.io/) for the web application framework
- [LangChain](https://python.langchain.com/) for LLM integration

- https://github.com/saksh-arch/Hallucination-Guard/blob/main/sakshi%20avhad_s%20Video%20-%20Nov%2021%2C%202025-VEED.mp3
