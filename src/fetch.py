from utils.spider import fetch_url
from utils.parser import parse_html

def main():
    url = "http://example.com"
    html_content = fetch_url(url)
    titles = parse_html(html_content)
    print("Titles found:", titles)

if __name__ == "__main__":
    main()
