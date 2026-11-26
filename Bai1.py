import re

def extract_numbers(text: str) -> list[str]:
    # \d : khớp với một chữ số (0-9)
    # +  : lặp lại 1 hoặc nhiều lần (gom các chữ số đứng cạnh nhau thành một số nguyên vẹn)
    pattern = r'\d+'
    return re.findall(pattern, text)

if __name__ == "__main__":
    s = 'My 2 favourite numbers are 8 and 25. My mobile is 0912203062'
    numbers = extract_numbers(s)
    
    print("Các số trích xuất được:", numbers) 