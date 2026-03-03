from security.storage import store_password, retrieve_password

master = input("Enter master password: ")
generated = input("Enter generated password to store: ")

store_password(master, generated)

print("\nTrying to retrieve...")

retrieved = retrieve_password("wrongPassword")

print("Decrypted password:", retrieved)