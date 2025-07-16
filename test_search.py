# import psycopg2

# conn = psycopg2.connect(
#     host="db.uvasrdqzqpsbfmcbisyg.supabase.co",
#     port=5432,
#     user="postgres",
#     password="Kingatbest@123",
#     dbname="postgres",
#     sslmode="require"
# )

# print("✅ Connected!")
# conn.close()
import psycopg2
import os

try:
    conn = psycopg2.connect("postgresql://postgres:Kingatbest123@db.uvasrdqzqpsbfmcbisyg.supabase.co:5432/postgres?sslmode=require")
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")