# AI Inventory Intelligence & Supplier Orchestration 📦🛡️🌐

An Intelligent Supplier Replenishment Platform that leverages Machine Learning, Generative AI, and Autonomous Agents to optimize demand forecasting, evaluate global supplier risk, and autonomously dispatch procurement requests.

## 🚀 Overview
This platform provides an end-to-end autonomous procurement suite, focusing on:
- **Demand Forecasting**: Predictive intelligence using rolling 365-day historical sales windows to extract core trends, seasonality multipliers, and demand volatility.
- **Supplier & Risk Intelligence**: Dynamically maps internal product categories to a Global Risk Database to rank vendors by a composite Risk Score (0-100) and negotiated discount potential.
- **Financial Intelligence**: Calculates "Inventory Optimization Impact" representing the financial risk of inaction (Stockout Risk vs. Holding Cost) and recommends optimized dynamic pricing.
- **Autonomous Outreach**: Integrated with SendGrid to generate and dispatch formatted, professional RFQ (Request for Quotation) emails to top-tier suppliers automatically.

## 🌐 Deployment Options
- **Streamlit Cloud (Live)**: Deploy easily via Streamlit Community Cloud.
- **Google Colab (Interactive Demo)**: Open the provided [AI_Inventory_Intelligence_Demo.ipynb](https://colab.research.google.com/github/lmudu2/ai-inventory-intelligence/blob/main/AI_Inventory_Intelligence_Demo.ipynb) directly in Google Colab to run the full dashboard & backend API in the cloud without installing anything locally.

## 🏗️ Architecture
- **Frontend**: Streamlit-based AI-first dashboard featuring premium glassmorphism aesthetics, dynamic metric visualizers, and an intuitive dual-view workflow (Global Market vs. Product Focus).
- **Core Orchestration**: LangGraph-driven state machines ensuring structured, sequential AI reasoning across data gathering, strategy formulation, and execution.
- **Data Engine**: Pandas and NumPy handling real-time data frame operations across 50,000+ mock SKUs, dense sales data, and global supply chain risk indices.
- **AI Brain**: Groq (Llama-3-70B-Versatile) integration simulating deep semantic category matching and synthesizing executive procurement proposals.

## 🛠️ Key Features
- **Intelligent Replenishment Engine**: Flags High-Demand products for urgent sourcing and Slow-Moving products for strategic liquidation.
- **Global Risk Mapping**: Evaluates suppliers on Geopolitical, Logistics, and Economic Volatility metrics to surface the "Best Overall" and "Best Price" candidates.
- **Semantic Product Bridge**: Employs LLMs to dynamically bridge the semantic gap between internal corporate SKUs and global vendor supply categories.
- **One-Click Execution**: Bypasses traditional procurement friction by allowing users to select AI-vetted suppliers and instantly trigger structured outreach.

## 🏁 Getting Started
1. **Clone the repository**:
   ```bash
   git clone https://github.com/lmudu2/ai-inventory-intelligence.git
   cd ai-inventory-intelligence
   ```
2. **Setup environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure API Keys**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SENDGRID_API_KEY=your_sendgrid_api_key_here
   SENDGRID_FROM_EMAIL=your_sender_email@example.com
   SENDGRID_TO_EMAIL=your_receiver_email@example.com
   NEWSDATA_API_KEY=your_newsdata_api_key_here
   ```
4. **Launch Platform**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Core Engineering Principles
This application was engineered with a focus on **Zero-Hardcoding Methodology**, **Actionable Financial Impact**, and **Autonomous Closed-Loop Execution** to transform raw data into immediate procurement leverage.

---
