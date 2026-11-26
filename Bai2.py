import re

def extract_emails(text: str) -> list[str]:
    # [a-zA-Z0-9._%+-]+ : phần tên người dùng (local part) trước dấu @
    # @                 : ký tự @ bắt buộc
    # [a-zA-Z0-9.-]+    : tên miền (domain, hỗ trợ cả subdomain như mail.edu)
    # \.[a-zA-Z]{2,}    : dấu chấm (escape bằng \.) theo sau bởi TLD có từ 2 chữ cái trở lên (.com, .vn, .edu,...)
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

if __name__ == "__main__":
    s = 'Hello from shubhamg199630@gmail.com to priya@yahoo.com     about the meeting @2PM'
    emails = extract_emails(s)

    print("Các email trích xuất được:", emails)