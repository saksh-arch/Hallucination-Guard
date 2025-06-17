"""
Hallucination Detector using uqlm library
"""
import streamlit as st

# Page config must be the very first Streamlit command
st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="wide"
)

# Now import all other libraries
import asyncio
import pandas as pd
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from uqlm import BlackBoxUQ, WhiteBoxUQ, LLMPanel, UQEnsemble
import os
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio to handle asyncio in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Default API key from .env file
default_api_key = os.getenv("GEMINI_API_KEY", "")

# Sidebar section for API key
st.sidebar.markdown("### 🔑 API Key")
api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:", 
    value="",
    type="password",
    help="Get your API key from https://aistudio.google.com/app/apikey"
)

# Set the API key in environment
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.sidebar.warning("⚠️ Please enter a Gemini API key to use this app")

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>🔍 Hallucination Detector</h1>", unsafe_allow_html=True)
st.write("""
This app demonstrates hallucination detection using the **uqlm** library with Google's Gemini models.
Select options in the sidebar to configure the hallucination detection method.
""")


# Sidebar title and API documentation link
st.sidebar.title("⚙️ Configuration")

# Model selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
    index=0
)

# Temperature setting
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Scorer method
scorer_method = st.sidebar.selectbox(
    "Hallucination Detection Method",
    ["Black-Box Scorer", "White-Box Scorer", "LLM-as-a-Judge", "Ensemble Scorer"]
)

# Method-specific options
if scorer_method == "Black-Box Scorer":
    bb_scorer = st.sidebar.selectbox(
        "Black-Box Scorer Type",
        ["semantic_negentropy", "exact_match", "noncontradiction", "all"]
    )
    num_responses = st.sidebar.slider("Number of responses", min_value=2, max_value=10, value=5)

elif scorer_method == "White-Box Scorer":
    wb_scorer = st.sidebar.selectbox(
        "White-Box Scorer Type",
        ["min_probability", "mean_probability", "perplexity", "all"]
    )

elif scorer_method == "LLM-as-a-Judge":
    judge_models = st.sidebar.multiselect(
        "Judge Models",
        ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
        default=["gemini-1.5-pro"]
    )
    if not judge_models:
        st.sidebar.warning("Please select at least one judge model")

elif scorer_method == "Ensemble Scorer":
    provide_ground_truth = st.sidebar.checkbox("Provide ground truth for calibration")

# Expandable info sections
with st.expander("🔍 About Hallucination Detection Methods"):
    st.markdown("""
    ### Black-Box Scorer
    Generates multiple responses for a prompt and compares their consistency using semantic similarity.
    
    ### White-Box Scorer
    Analyzes token-by-token probabilities to determine how confident the model is in its outputs.
    
    ### LLM-as-a-Judge
    Uses other LLMs to evaluate whether the output of the primary LLM contains hallucinations.
    
    ### Ensemble Scorer
    Combines multiple scoring methods for a more robust evaluation.
    """)

# Input section
st.markdown("<h2 class='sub-header'>📝 Input</h2>", unsafe_allow_html=True)

# Example prompts
example_prompts = {
    "Factual query": "What is the capital of France?",
    "Opinion query": "What's the best programming language and why?",
    "Fictional entity": "Tell me about the history of Atlantis.",
    "Mathematical reasoning": "What is the result of dividing 1 by 0?",
    "Custom prompt": ""
}

prompt_type = st.selectbox("Select prompt type or create your own:", list(example_prompts.keys()))
if prompt_type == "Custom prompt":
    prompt = st.text_area("Enter your custom prompt:", height=100)
else:
    prompt = st.text_area("Prompt:", value=example_prompts[prompt_type], height=100)

# For ensemble scorer with ground truth
ground_truth = None
if scorer_method == "Ensemble Scorer" and provide_ground_truth:
    ground_truth = st.text_input("Ground truth answer:")

# Create full prompt
full_prompt = prompt.strip()

# Function to create and run the appropriate scorer
async def run_scorer(scorer_method, prompt, model, temperature):
    """Run the selected hallucination scorer"""
    try:
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=2048,
            timeout=60,
            max_retries=3
        )
        
        # Run appropriate scorer
        if scorer_method == "Black-Box Scorer":
            scorers = [bb_scorer] if bb_scorer != "all" else ["semantic_negentropy", "exact_match", "noncontradiction"]
            st.text(f"Using {scorers} for Black-Box scoring...")
            
            scorer = BlackBoxUQ(llm=llm, scorers=scorers, use_best=True)
            results = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses)
            
        elif scorer_method == "White-Box Scorer":
            scorers = [wb_scorer] if wb_scorer != "all" else ["min_probability", "mean_probability", "perplexity"]
            st.text(f"Using {scorers} for White-Box scoring...")
            
            scorer = WhiteBoxUQ(llm=llm, scorers=scorers)
            results = await scorer.generate_and_score(prompts=[prompt])
            
        elif scorer_method == "LLM-as-a-Judge":
            if not judge_models:
                st.error("Please select at least one judge model")
                return None
                
            judges = [ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature
            ) for model_name in judge_models]
            
            st.text(f"Using {len(judges)} judge models...")
            scorer = LLMPanel(llm=llm, judges=judges)
            results = await scorer.generate_and_score(prompts=[prompt])
            
        else:  # Ensemble Scorer
            scorers = [
                "exact_match", "noncontradiction",  # black-box scorers
                "min_probability",  # white-box scorer
                llm  # use same LLM as a judge
            ]
            st.text("Using ensemble of scorers...")
            scorer = UQEnsemble(llm=llm, scorers=scorers)
            
            if ground_truth:
                st.text("Tuning ensemble with ground truth...")
                await scorer.tune(prompts=[prompt], ground_truth_answers=[ground_truth])
                
            results = await scorer.generate_and_score(prompts=[prompt])
        
        # Return dataframe with results
        return results.to_df()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Execute button
if st.button("Generate and Score"):
    if not prompt:
        st.warning("Please enter a prompt.")
    elif not api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar")
    else:
        with st.spinner("Generating response and calculating hallucination scores..."):
            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the scorer
            df = loop.run_until_complete(run_scorer(scorer_method, full_prompt, model_option, temperature))
            
            # Display results
            if df is not None and not df.empty:
                st.markdown("<h2 class='sub-header'>Results</h2>", unsafe_allow_html=True)
                
                # Show the response
                response = df['response'].iloc[0]
                st.markdown("<h3>LLM Response:</h3>", unsafe_allow_html=True)
                st.write(response)
                
                # Show scores
                st.markdown("<h3>Confidence Scores:</h3>", unsafe_allow_html=True)
                
                # Extract scores
                scores_dict = {}
                for col in df.columns:
                    # Check for standard score columns and also specific uqlm score columns
                    if (col.endswith('_score') or "_score" in col or 
                        col in ['semantic_negentropy', 'exact_match', 'noncontradiction', 
                               'min_probability', 'mean_probability', 'perplexity']):
                        score_name = col
                        if not pd.isna(df[col].iloc[0]):
                            score_value = float(df[col].iloc[0])
                            scores_dict[score_name] = score_value
                
                # Display scores as a bar chart
                if scores_dict:
                    scores_df = pd.DataFrame({
                        'Scorer': list(scores_dict.keys()),
                        'Confidence Score': list(scores_dict.values())
                    })
                    
                    fig = px.bar(
                        scores_df, 
                        x='Scorer', 
                        y='Confidence Score',
                        color='Confidence Score',
                        color_continuous_scale='RdYlGn',
                        range_color=[0, 1],
                        height=400,
                        title='Hallucination Confidence Scores (Higher = Less Likely to be Hallucinating)'
                    )
                    
                    fig.update_layout(
                        xaxis_title='Scorer Type',
                        yaxis_title='Confidence Score (0-1)',
                        yaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpreting the score
                    avg_score = sum(scores_dict.values()) / len(scores_dict)
                    
                    if avg_score > 0.8:
                        st.success(f"✅ Average confidence score: {avg_score:.2f} - The response is likely factual.")
                    elif avg_score > 0.5:
                        st.warning(f"⚠️ Average confidence score: {avg_score:.2f} - The response may contain some inaccuracies.")
                    else:
                        st.error(f"❌ Average confidence score: {avg_score:.2f} - The response likely contains hallucinations.")
                    
                    # Show raw data
                    with st.expander("Show raw data"):
                        st.dataframe(df)
                else:
                    st.error("No scores were calculated.")
                    st.dataframe(df)
            else:
                st.error("Error generating results. Please try a different configuration.")

# Footer with explanation
st.markdown("---")
with st.expander("📚 About uqlm Library"):
    st.markdown("""
    The **uqlm** (Uncertainty Quantification for Language Models) library provides tools to quantify the uncertainty
    or confidence in language model outputs, helping to detect potential hallucinations.
    
    Learn more about the methods:
    
    - **Black-Box**: Generates multiple responses and measures their consistency
    - **White-Box**: Analyzes token probabilities from the model
    - **LLM-as-a-Judge**: Uses other LLMs to evaluate the primary model's outputs
    - **Ensemble**: Combines multiple methods for more robust evaluation
    
    Higher scores (closer to 1.0) indicate higher confidence and lower likelihood of hallucination.
    """)

# Add demo guidance
with st.expander("🔬 Demonstration Tips"):
    st.markdown("""
    ### Tips for demonstrating hallucination detection:
    
    1. **Compare factual vs fictional questions**: Try a factual question like "What is the capital of France?" and a fictional one like "Tell me about Zorblaxians from Planet Xenu".
    
    2. **Test mathematical reasoning**: The model should be confident about simple math like "1 + 2 + 3 + 4 + 5" but less confident about impossible questions like "1 / 0".
    
    3. **Try different scorer methods**: Black-Box scorers work best with multiple samples, while White-Box is faster but may have less nuance.
    
    4. **Adjust temperature**: Higher temperature values lead to more creative (but potentially more hallucinated) responses.
    """)
