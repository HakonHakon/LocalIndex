from vector_db import LocalVectorDBChat
import os

def print_help():
    print("\nAvailable commands:")
    print("  train          - Add new documents")
    print("  list           - List all stored documents")
    print("  categories     - List all categories")
    print("  list <category>- List documents in a specific category")
    print("  help           - Show this help message")
    print("  quit           - Exit the program")
    print("  Any other input will be treated as a question")

def main():
    chat_system = LocalVectorDBChat(use_persistent=True)
    
    print("Welcome to Local Vector DB Chat!")
    print("--------------------------------")
    print_help()
    
    while True:
        command = input("\nEnter command or question: ").strip()
        
        if command.lower() == 'quit':
            break
            
        elif command.lower() == 'help':
            print_help()
            
        elif command.lower() == 'train':
            category = input("Enter category (or press Enter for 'general'): ").strip()
            if not category:
                category = "general"
                
            print(f"Enter documents for category '{category}' (one per line). Enter an empty line to finish:")
            documents = []
            while True:
                text = input().strip()
                if not text:
                    break
                documents.append({"text": text})
            
            if documents:
                chat_system.train(documents, category)
            
        elif command.lower() == 'list':
            documents = chat_system.list_documents()
            print("\nStored documents:")
            for doc in documents:
                print(f"\nID: {doc['id']}")
                print(f"Category: {doc['category']}")
                print(f"Date: {doc['timestamp']}")
                print(f"Preview: {doc['text_preview']}")
                
        elif command.lower().startswith('list '):
            category = command[5:].strip()
            documents = chat_system.list_documents(category)
            print(f"\nDocuments in category '{category}':")
            for doc in documents:
                print(f"\nID: {doc['id']}")
                print(f"Date: {doc['timestamp']}")
                print(f"Preview: {doc['text_preview']}")
                
        elif command.lower() == 'categories':
            categories = chat_system.list_categories()
            print("\nAvailable categories:")
            for category in categories:
                docs = chat_system.list_documents(category)
                print(f"  {category} ({len(docs)} documents)")
            
        else:
            try:
                response = chat_system.chat(command)
                print("\nResponse:", response)
            except Exception as e:
                print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()