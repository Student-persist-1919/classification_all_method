import re

def extract_emails(text: str) -> list[str]:
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

if __name__ == "__main__":
    s = 'Hello from shubhamg199630@gmail.com to priya@yahoo.com     about the meeting @2PM'
    emails = extract_emails(s)

    print("Các email trích xuất được:", emails)