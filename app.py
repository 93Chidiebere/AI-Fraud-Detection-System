import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# App title
st.set_page_config(page_title="Fraud Detection System", page_icon="🚨", layout="wide")
st.title("🚨 AI-Powered Fraud Detection System")
st.markdown("**Real-time transaction analysis for Nigerian banks**")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('xgb_fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_models()

# Sidebar
st.sidebar.header("Transaction Details")
st.sidebar.markdown("Enter transaction information below:")

# Input form
with st.sidebar:
    transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
    amount = st.number_input("Amount (₦)", min_value=0.0, value=50000.0, step=1000.0)
    
    st.markdown("#### Origin Account")
    origin_balance = st.number_input("Origin Balance (₦)", min_value=0.0, value=100000.0)
    
    st.markdown("#### Destination Account")
    dest_balance = st.number_input("Destination Balance (₦)", min_value=0.0, value=0.0)
    dest_in_degree = st.slider("Destination Incoming Connections", 0, 100, 5)
    
    analyze_button = st.button("🔍 Analyze Transaction", type="primary")

# Main content
col1, col2 = st.columns([1, 1])

if analyze_button:
    
    # Validation 1: Insufficient funds
    if amount > origin_balance:
        with col1:
            st.error("⛔ **TRANSACTION REJECTED**")
            st.markdown(f"""
            ### Insufficient Funds
            - **Account Balance:** ₦{origin_balance:,.2f}
            - **Transaction Amount:** ₦{amount:,.2f}
            - **Deficit:** ₦{(amount - origin_balance):,.2f}
            """)
        
        with col2:
            st.markdown("### 🚨 Critical Alert")
            st.markdown("""
            **This is a HIGH RISK fraud indicator:**
            - 🔴 Attempting to overdraw account
            - 🔴 Balance manipulation attempt
            - 🔴 Possible account takeover
            
            **Recommended Actions:**
            - ❌ Block transaction immediately
            - 📞 Contact account holder
            - 🔒 Freeze account pending investigation
            """)
        st.stop()  # Stop here - don't run model
    
    # Validation 2: Invalid amounts
    if amount <= 0:
        st.error("⛔ Invalid transaction amount. Must be greater than zero.")
        st.stop()
    
    # Validation 3: Negative balances
    if origin_balance < 0 or dest_balance < 0:
        st.error("⛔ Invalid account balance. Balances cannot be negative.")
        st.stop()
    
    
    # Calculate features
    origin_balance_change = amount
    dest_balance_change = amount
    origin_error = 1 if (origin_balance - amount) < 0 else 0
    dest_error = 0
    origin_zero_after = 1 if (origin_balance - amount) == 0 else 0
    dest_zero_before = 1 if dest_balance == 0 else 0
    amount_to_origin = amount / (origin_balance + 1)
    amount_to_dest = amount / (dest_balance + 1)
    
    # Create feature df
    # features = np.array([[
    #     0,  # step (not used in demo)
    #     amount,
    #     origin_balance_change,
    #     dest_balance_change,
    #     origin_error,
    #     dest_error,
    #     origin_zero_after,
    #     dest_zero_before,
    #     amount_to_origin,
    #     amount_to_dest,
    #     1,  # origin_out_degree
    #     dest_in_degree,
    #     0.001,  # origin_pagerank
    #     0.001,  # dest_pagerank
    #     0.5  # velocity
    # ]])
    
    features = pd.DataFrame({
    'step': [0],
    'amount': [amount],
    'origin_balance_change': [origin_balance_change],
    'destination_balance_change': [dest_balance_change],
    'origin_error': [origin_error],
    'destination_error': [dest_error],
    'origin_zero_after': [origin_zero_after],
    'destination_zero_before': [dest_zero_before],
    'amount_to_origin_balance': [amount_to_origin],
    'amount_to_destination_balance': [amount_to_dest],
    'origin_out_degree': [1],
    'destination_in_degree': [dest_in_degree],
    'origin_pagerank': [0.001],
    'destination_pagerank': [0.001],
    'velocity': [0.5]
})
    
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    fraud_prob = model.predict_proba(features_scaled)[0][1]
    is_fraud = model.predict(features_scaled)[0]
    
    # Display results
    with col1:
        st.markdown("### 📊 Analysis Results")
        
        # Risk score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if fraud_prob > 0.5 else "orange" if fraud_prob > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Decision
        if fraud_prob > 0.5:
            st.error(f"⚠️ **HIGH RISK**: {fraud_prob*100:.1f}% fraud probability")
            st.markdown("**Recommendation:** Block transaction and investigate")
        elif fraud_prob > 0.3:
            st.warning(f"⚠️ **MEDIUM RISK**: {fraud_prob*100:.1f}% fraud probability")
            st.markdown("**Recommendation:** Require additional authentication")
        else:
            st.success(f"✅ **LOW RISK**: {fraud_prob*100:.1f}% fraud probability")
            st.markdown("**Recommendation:** Approve transaction")
    
    # with col2:
    #     st.markdown("### 🎯 Risk Factors")
        
    #     risk_factors = []
    #     if origin_zero_after:
    #         risk_factors.append("🔴 Origin account will be emptied")
    #     if dest_zero_before:
    #         risk_factors.append("🔴 Destination account is new (zero balance)")
    #     if amount_to_origin > 0.9:
    #         risk_factors.append("🔴 Transaction is >90% of origin balance")
    #     if dest_in_degree < 2:
    #         risk_factors.append("🟡 Destination account has few incoming connections")
    #     if amount > 200000:
    #         risk_factors.append("🟡 Large transaction amount")
        
    #     if risk_factors:
    #         for factor in risk_factors:
    #             st.markdown(f"- {factor}")
    #     else:
    #         st.markdown("✅ No major risk factors detected")
        
    #     # Transaction summary
    #     st.markdown("### 📋 Transaction Summary")
    #     st.markdown(f"""
    #     - **Type:** {transaction_type}
    #     - **Amount:** ₦{amount:,.2f}
    #     - **Origin Balance:** ₦{origin_balance:,.2f}
    #     - **Destination Balance:** ₦{dest_balance:,.2f}
    #     - **Network Connections:** {dest_in_degree}
    #     """)

    with col2:
        st.markdown("### 🎯 Risk Factors")
        
        risk_factors = []
        
        # 🆕 CHECK 1: INSUFFICIENT FUNDS (CRITICAL!)
        if amount > origin_balance:
            risk_factors.append("🔴 INSUFFICIENT FUNDS - Transaction exceeds account balance!")
            risk_factors.append(f"   └─ Deficit: ₦{(amount - origin_balance):,.2f}")
        
        # CHECK 2: ORIGIN WILL BE EMPTIED
        if origin_zero_after:
            risk_factors.append("🔴 Origin account will be emptied")
        
        # CHECK 3: NEW DESTINATION ACCOUNT
        if dest_zero_before:
            risk_factors.append("🔴 Destination account is new (zero balance)")
        
        # CHECK 4: HIGH PERCENTAGE OF BALANCE (only if not already flagged for insufficient funds)
        if amount_to_origin > 0.9 and amount <= origin_balance:
            risk_factors.append(f"🔴 Transaction is {amount_to_origin*100:.1f}% of origin balance")
        
        # CHECK 5: LOW DESTINATION CONNECTIONS
        if dest_in_degree < 2:
            risk_factors.append("🟡 Destination account has few incoming connections")
        
        # CHECK 6: LARGE AMOUNT
        if amount > 200000:
            risk_factors.append(f"🟡 Large transaction amount (₦{amount:,.0f})")
        
        # 🆕 CHECK 7: CALCULATION ERROR DETECTED
        if origin_error:
            risk_factors.append("⚠️ Balance calculation error detected (possible fraud attempt)")
        
        # Display risk factors
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("✅ No major risk factors detected")

            # Transaction summary
            st.markdown("### 📋 Transaction Summary")
            st.markdown(f"""
            - **Type:** {transaction_type}
            - **Amount:** ₦{amount:,.2f}
            - **Origin Balance:** ₦{origin_balance:,.2f}
            - **Destination Balance:** ₦{dest_balance:,.2f}
            - **Network Connections:** {dest_in_degree}
            """)
        
        
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 15px;'>
    <p>Built for comparing general purpose ML algorithms with 
        <strong>TimeCost Gradient Machine</strong> in combating Digital Fraud</p>
    <p>Model: <strong>XGBoost</strong> | Precision: <strong>0.98</strong> | 
       F1-Score: <strong>0.99</strong></p>
</div>
""", unsafe_allow_html=True)