import streamlit as st
import sqlite3
import pandas as pd
from collections import defaultdict
import hashlib
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import unicodedata

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Database Functions
# -----------------------------

def create_connection():
    conn = sqlite3.connect('workload.db', check_same_thread=False)
    return conn

def setup_database():
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS staff (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            groups TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            items TEXT NOT NULL,
            assigned_to TEXT,
            status TEXT DEFAULT 'WIP'
        )
    """)

    conn.commit()
    conn.close()

def get_active_groups():
    # Logic to fetch active groups
    return ["Veg Pizza", "NV Pizza", "Sandwich"]

def place_order(customer_name, items):
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Insert the new order into the database
    cursor.execute("""
        INSERT INTO orders (customer_name, items)
        VALUES (?, ?)
    """, (customer_name, ",".join(items)))

    conn.commit()
    conn.close()

def authenticate(staff_name, staff_password):
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Hash the input password
    hashed_password = hashlib.sha256(staff_password.encode()).hexdigest()

    # Check if the staff exists with the given credentials
    cursor.execute("""
        SELECT * FROM staff WHERE name = ? AND password = ?
    """, (staff_name, hashed_password))
    result = cursor.fetchone()
    conn.close()

    return result is not None

def auto_assign_orders():
    # Placeholder for auto-assigning orders
    pass

def get_staff_orders(staff_name):
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch orders assigned to the staff member
    cursor.execute("""
        SELECT id, customer_name, items, status FROM orders WHERE assigned_to = ?
    """, (staff_name,))
    rows = cursor.fetchall()
    conn.close()

    return rows

def complete_order(order_id):
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Update the order status to "Completed"
    cursor.execute("""
        UPDATE orders SET status = 'Completed' WHERE id = ?
    """, (order_id,))

    conn.commit()
    conn.close()

def get_dashboard_data():
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch all orders
    cursor.execute("""
        SELECT id, assigned_to, status FROM orders
    """)
    rows = cursor.fetchall()
    conn.close()

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["order_id", "assigned_to", "status"])
    return df

def add_staff(name, password, groups):
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Hash the password for security
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Insert the new staff member into the database
    cursor.execute("""
        INSERT INTO staff (name, password, groups)
        VALUES (?, ?, ?)
    """, (name, hashed_password, groups))

    conn.commit()
    conn.close()

def edit_staff(name, groups):
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Update the staff member's groups
    cursor.execute("""
        UPDATE staff SET groups = ? WHERE name = ?
    """, (groups, name))

    conn.commit()
    conn.close()

def delete_staff(name):
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Delete the staff member from the database
    cursor.execute("""
        DELETE FROM staff WHERE name = ?
    """, (name,))

    conn.commit()
    conn.close()

def get_all_staff():
    # Connect to the database
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch all staff members
    cursor.execute("SELECT name, groups FROM staff")
    rows = cursor.fetchall()
    conn.close()

    # Convert to DataFrame
    staff_df = pd.DataFrame(rows, columns=["staff_name", "groups"])
    return staff_df

# -----------------------------
# AI Functions with Groq
# -----------------------------

def load_groq_model():
    # Fetch the Groq API Key from the .env file
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
    if groq_api_key:
        return ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")
    return None

def recommend_items(customer_name, current_items, groq_llm):
    if not groq_llm:
        return []

    prompt = PromptTemplate(
        input_variables=["customer_name", "current_items"],
        template="""You are an expert in customer preferences.
Given the customer name: {customer_name}
And their current order: {current_items}
Suggest 2 complementary items they might like."""
    )

    query = prompt.format(customer_name=customer_name, current_items=", ".join(current_items))
    recommendation = groq_llm.predict(query)
    return [item.strip() for item in recommendation.split(",")]

def smart_assign_staff(order_item, available_staff, groq_llm):
    if not groq_llm or not available_staff:
        return None

    # Fetch pending orders for each staff member
    staff_pending_orders = defaultdict(int)
    for staff_name, _ in available_staff.items():
        staff_orders = get_staff_orders(staff_name)
        staff_pending_orders[staff_name] = sum(1 for _, _, _, status in staff_orders if status == "WIP")

    # Format staff info with pending orders
    staff_info = ", ".join([
        f"{name}: Skills({skills}), Pending Orders({staff_pending_orders[name]})"
        for name, skills in available_staff.items()
    ])

    prompt = PromptTemplate(
        input_variables=["order_item", "staff_info"],
        template="""You are an expert workload optimizer.
Given the order item: {order_item}
And available staff with their skill groups and pending orders: {staff_info}
Suggest only ONE best staff name to assign this order to, considering workload balance and skill matching."""
    )

    query = prompt.format(order_item=order_item, staff_info=staff_info)
    assignment = groq_llm.predict(query)
    return assignment.strip()

def analyze_trends(df, groq_llm):
    if not groq_llm:
        return "No AI model available."

    prompt = PromptTemplate(
        input_variables=["data_summary"],
        template="""You are an expert in analyzing workload trends.
Given the following summary of orders: {data_summary}
Provide insights into trends and predictions for future orders."""
    )

    data_summary = (
        f"Total Orders: {df.shape[0]}, "
        f"In Progress: {df[df['status']=='WIP'].shape[0]}, "
        f"Completed: {df[df['status']=='Completed'].shape[0]}"
    )
    query = prompt.format(data_summary=data_summary)
    insights = groq_llm.predict(query)
    return insights

def suggest_groups_for_staff(groq_llm):
    if not groq_llm:
        return []

    prompt = PromptTemplate(
        input_variables=[],
        template="""You are an expert in workload optimization.
Suggest 2 optimal skill groups for a new staff member based on current workload trends."""
    )

    query = prompt.format()
    suggestions = groq_llm.predict(query)
    return [group.strip() for group in suggestions.split(",")]

def staff_dashboard_message(staff_name, groq_llm, orders_pending):
    if not groq_llm:
        return ""

    prompt = PromptTemplate(
        input_variables=["staff_name", "orders_pending"],
        template="""You are an assistant helping {staff_name}.
Summarize politely their pending orders list: {orders_pending}
"""
    )

    query = prompt.format(staff_name=staff_name, orders_pending=", ".join(orders_pending))
    message = groq_llm.predict(query)
    return message

# -----------------------------
# Static Data
# -----------------------------

product_group_mapping = {
    'Veg Pizza': 'Veg Pizza',
    'NV Pizza': 'NV Pizza',
    'Sandwich': 'Sandwich',
    'Burger': 'Burger',
    'Coke': 'Drinks'
}

# -----------------------------
# Streamlit App
# -----------------------------

def clean_text(text):
    # Normalize the text to standard Unicode form
    normalized_text = unicodedata.normalize('NFKC', text)
    # Remove invalid surrogate pairs
    cleaned_text = ''.join(c for c in normalized_text if unicodedata.category(c) != 'Cs')
    return cleaned_text

st.set_page_config(page_title="Workload Management System", layout="wide")
st.title(clean_text("📦 Workload Management System"))

setup_database()

groq_llm = load_groq_model()

menu = ["Place Order", "Staff Login", "Dashboard", "Admin Panel"]
choice = st.sidebar.selectbox("Select Action", menu)

if choice == "Place Order":
    st.subheader(clean_text("📝 Place a New Order"))
    customer_name = st.text_input("Customer Name")
    active_groups = get_active_groups()
    available_items = [item for item, group in product_group_mapping.items() if group in active_groups]

    items = st.multiselect("Select Available Items", available_items)
    
    # AI Recommendations
    if items and groq_llm:
        recommended_items = recommend_items(customer_name, items, groq_llm)
        st.write("### AI Recommendations:")
        st.write(", ".join(recommended_items))

    if st.button("Place Order"):
        if customer_name and items:
            place_order(customer_name, items)
            st.success("Order placed successfully!")
        else:
            st.error("Please enter Customer Name and select at least one item.")

elif choice == "Staff Login":
    st.subheader(clean_text("👨‍💻 Staff Login"))
    staff_name = st.text_input("Staff Name")
    staff_password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(staff_name, staff_password):
            st.success(f"Welcome {staff_name}!")
            st.session_state.logged_in = staff_name
        else:
            st.error("Invalid Credentials!")

    if 'logged_in' in st.session_state:
        auto_assign_orders()
        staff_orders = get_staff_orders(st.session_state.logged_in)

        if groq_llm:
            order_list = [item[2] for item in staff_orders]
            message = staff_dashboard_message(st.session_state.logged_in, groq_llm, order_list)
            st.info(message)

        st.write("### Your Assigned Orders")
        for order in staff_orders:
            order_id, order_number, item, status = order
            st.write(f"**{order_number}** - {item} - Status: {status}")
            if st.button(f"Mark Complete {order_id}"):
                complete_order(order_id)
                st.success("Order marked as completed!")

elif choice == "Dashboard":
    st.subheader(clean_text("📊 Dashboard"))
    df = get_dashboard_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", df.shape[0])
    with col2:
        st.metric("In Progress", df[df['status']=='WIP'].shape[0])
    with col3:
        st.metric("Completed", df[df['status']=='Completed'].shape[0])

    # AI Insights
    if groq_llm:
        insights = analyze_trends(df, groq_llm)
        st.write("### AI Insights:")
        st.info(insights)

    st.write("### Staff Performance")
    staff_perf = df[df['status']=='Completed'].groupby('assigned_to').size().reset_index(name='Completed Orders')
    st.bar_chart(staff_perf.set_index('assigned_to'))

    st.write("### All Orders")
    st.dataframe(df)

elif choice == "Admin Panel":
    st.subheader(clean_text("🛠️ Admin Panel"))

    admin_choice = st.radio("Action", ["Add Staff", "Edit Staff", "Delete Staff", "View Staff"])

    if admin_choice == "Add Staff":
        name = st.text_input("Staff Name")
        password = st.text_input("Password", type="password")

        # AI Suggestions
        if groq_llm:
            suggested_groups = suggest_groups_for_staff(groq_llm)
            st.write("### AI-Suggested Groups:")
            st.write(", ".join(suggested_groups))

        groups = st.text_input("Groups (comma separated)")
        if st.button("Add Staff"):
            if name and password and groups:
                add_staff(name, password, groups)
                st.success("Staff added successfully!")
            else:
                st.error("All fields are required!")

    elif admin_choice == "Edit Staff":
        name = st.text_input("Staff Name to Edit")
        groups = st.text_input("New Groups (comma separated)")
        if st.button("Update Staff"):
            if name and groups:
                edit_staff(name, groups)
                st.success("Staff updated successfully!")
            else:
                st.error("All fields are required!")

    elif admin_choice == "Delete Staff":
        name = st.text_input("Staff Name to Delete")
        if st.button("Delete Staff"):
            if name:
                delete_staff(name)
                st.success("Staff deleted successfully!")
            else:
                st.error("Staff Name is required!")

    elif admin_choice == "View Staff":
        st.write("### Current Staff")
        staff_df = get_all_staff()
        st.dataframe(staff_df)