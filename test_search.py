# auth_test.py
from auth import sign_up_user, login_user

email = "dvaghani@umd.edu.com"
password = "strongpassword123"

# Signup (only once)
# sign_up_user(email, password)

# Login
session = sign_up_user(email, password)
print(session)
