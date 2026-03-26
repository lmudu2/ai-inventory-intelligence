import pandas as pd
import numpy as np
import sqlite3
import os
import requests
from groq import Groq

# API Keys from User
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

if not GROQ_API_KEY or not NEWSDATA_API_KEY:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


class SupplyChainAgent:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.products_df = pd.read_csv(os.path.join(data_dir, "products_50k.csv"))
        self.sales_df = pd.read_csv(os.path.join(data_dir, "sales_dense.csv"))
        self.risk_df = pd.read_csv(os.path.join(data_dir, "supply_chain_risk_analysis.csv"))
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
    def get_high_demand_products(self, threshold=100):
        """
        Identify products with high demand based on recent sales.
        For simplicity, we'll sum up the quantity sold in the sales_dense dataset.
        """
        if 'product_id' not in self.sales_df.columns and 'sku' in self.sales_df.columns:
            self.sales_df = self.sales_df.rename(columns={'sku': 'product_id'})
        if 'product_id' not in self.products_df.columns and 'sku' in self.products_df.columns:
            self.products_df = self.products_df.rename(columns={'sku': 'product_id'})
        
        # Consistent format: Ensure numeric IDs like '1' become 'SKU-000001'
        def format_sku(val):
            val_str = str(val).strip()
            if val_str.isdigit():
                return f"SKU-{int(val_str):06d}"
            return val_str

        self.sales_df['product_id'] = self.sales_df['product_id'].apply(format_sku)
        self.products_df['product_id'] = self.products_df['product_id'].apply(format_sku)
            
        demand = self.sales_df.groupby('product_id')['quantity_sold'].sum().reset_index()
        high_demand = demand[demand['quantity_sold'] > threshold]
        
        # Merge with product names
        result = high_demand.merge(self.products_df[['product_id', 'name', 'category', 'unit_price']], on='product_id')
        return result.sort_values(by='quantity_sold', ascending=False)

    def research_manufacturers(self, product_category):
        """
        Find manufacturers for a specific category, filtering by risk and price.
        """
        # In the risk dataset, 'Product' column contains the item category/type
        # We also need to map available columns for research
        # Available: Order_ID, Product, Vendor, Region, Country, Shipment_Mode, Order_Value_USD, Country_Risk_Index...
        
        suppliers = self.risk_df[self.risk_df['Product'].str.contains(product_category, case=False, na=False)]
        
        # If no direct match by product, show top suppliers in the region for that category's general type
        if suppliers.empty:
            suppliers = self.risk_df.head(20) # Fallback to top suppliers

        suppliers = suppliers.copy()
        
        # Rename columns to match expected research output
        suppliers = suppliers.rename(columns={
            'Vendor': 'Supplier Name',
            'Country': 'Location',
            'Country_Risk_Index': 'Risk Score',
            'Order_Value_USD': 'Base Price'
        })
        
        # Mock additional fields for the UI
        suppliers['Discount %'] = np.random.choice([0, 5, 10, 15, 20], size=len(suppliers))
        suppliers['Final Price'] = suppliers['Base Price'] * (1 - suppliers['Discount %']/100)
        suppliers['Delay Probability'] = np.random.uniform(0.01, 0.15, size=len(suppliers))
        suppliers['Current Stock Levels'] = np.random.randint(100, 5000, size=len(suppliers))
        
        research_results = suppliers[[
            'Supplier Name', 'Location', 'Risk Score', 'Delay Probability', 
            'Base Price', 'Discount %', 'Final Price', 'Current Stock Levels'
        ]]
        
        return research_results.sort_values(by=['Risk Score', 'Final Price'])

    def generate_proposal(self, product_id, supplier_name):
        """
        Generate a structured proposal for procurement.
        """
        product = self.products_df[self.products_df['product_id'] == product_id].iloc[0]
        # In risk_df, the supplier name is in 'Vendor'
        supplier = self.risk_df[self.risk_df['Vendor'] == supplier_name].iloc[0]
        
        proposal = {
            "Product": product['name'],
            "Category": product['category'],
            "Supplier": supplier_name,
            "Location": supplier['Country'],
            "Risk Assessment": {
                "Score": supplier['Country_Risk_Index'],
                "Logistics Mode": supplier['Shipment_Mode']
            },
            "Action": "Submit Request for Quotation (RFQ)"
        }
        return proposal

    def get_market_news(self, query):
        """
        Fetch real-time news for a product or category using NewsData.io API.
        """
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&language=en"
        try:
            response = requests.get(url)
            data = response.json()
            if data.get("status") == "success":
                return data.get("results", [])[:5] # Return top 5 news items
            return []
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def llm_generate_proposal(self, product_info, supplier_info, news_context=None):
        """
        Use Groq (Llama 3) to synthesize a professional procurement proposal.
        """
        news_text = ""
        if news_context:
            news_text = "\nRecent Market Trends:\n" + "\n".join([f"- {n['title']}: {n['description']}" for n in news_context if n.get('description')])

        prompt = f"""
        You are an AI Supply Chain Specialist. Generate a professional procurement proposal based on the following data:

        PRODUCT DETAILS:
        - Name: {product_info['name']}
        - Category: {product_info['category']}
        - Quantity Sold in Last Month: {product_info['quantity_sold']}
        
        SUPPLIER DETAILS:
        - Name: {supplier_info['Supplier Name']}
        - Location: {supplier_info['Location']}
        - Risk Score: {supplier_info['Risk Score']}
        - Price Highlight: {supplier_info['Final Price']} USD
        
        {news_text}

        Your goal is to justify why we should stock up on this product and why this specific supplier is the best choice.
        Be professional, concise, and persuasive. Format the output as a formal memo.
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional supply chain procurement agent."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating proposal via LLM: {e}"

if __name__ == "__main__":
    agent = SupplyChainAgent()
    print("--- High Demand Products ---")
    high_demand = agent.get_high_demand_products(threshold=150)
    print(high_demand.head())
    
    if not high_demand.empty:
        category = high_demand.iloc[0]['category']
        print(f"\n--- Researching Manufacturers for Category: {category} ---")
        manufacturers = agent.research_manufacturers(category)
        print(manufacturers.head())
        
        if not manufacturers.empty:
            print("\n--- Generating Proposal Case ---")
            proposal = agent.generate_proposal(high_demand.iloc[0]['product_id'], manufacturers.iloc[0]['Supplier Name'])
            print(proposal)
