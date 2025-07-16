# from supabase import create_client
# import os
# from dotenv import load_dotenv
# load_dotenv()
# url = os.getenv("SUPABASE_URL")
# key = os.getenv("SUPABASE_KEY")
# supabase = create_client(url, key)


# class SupabaseMemory:
#     def __init__(self):
#         self.table = "chat_messages"

#     def put(self, thread_id: str, role: str, content: str):
#         supabase.table(self.table).insert({
#             "thread_id": thread_id,
#             "role": role,
#             "content": content
#         }).execute()

#     def get_last_messages(self, thread_id: str, limit=10):
#         response = supabase.table(self.table) \
#             .select("*") \
#             .eq("thread_id", thread_id) \
#             .order("timestamp", desc=False) \
#             .limit(limit) \
#             .execute()

#         return response.data
