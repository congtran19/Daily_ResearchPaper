from bs4 import BeautifulSoup
import requests
from data.models import Paper
BASE_URL = 'https://arxiv.org/list/cs.AI/new'
class ArxivSource:
    def __init__(self,BASE_URL= 'https://arxiv.org/list/cs.AI/new'):
        self.BASE_URL = BASE_URL

    def fetch_latest(self,n= 5): #Lay 5 bai bao
        print(f"Dang crawl data tu {self.BASE_URL} ...")
        response = requests.get(self.BASE_URL)
        if response.status_code !=200:
            raise Exception ("Cannot access arXiv")
    
        soup = BeautifulSoup(response.text,'html.parser')
        paper = []
        item = soup.select_one('dt')
        articles = soup.find('dl',id ='articles')
        dts = articles.find_all("dt")
        dds = articles.find_all("dd")
        # 5. Duyệt qua 5 bài đầu tiên
        papers = []
        for dt, dd in zip(dts[:n], dds[:n]):
            # Lấy link bài báo
            link_tag = dt.find("a", href=True)
            link = "https://arxiv.org" + link_tag["href"] if link_tag else None

            # Lấy title, authors, abstract
            title_tag = dd.find("div", class_="list-title")
            author_tag = dd.find("div", class_="list-authors")
            abstract_tag = dd.find("p")

            title = title_tag.text.replace("Title:", "").strip() if title_tag else ""
            authors = author_tag.text.replace("Authors:", "").strip() if author_tag else ""
            abstract = abstract_tag.text.strip() if abstract_tag else ""

            papers.append(Paper(
                title=title,
                authors=[a.strip() for a in authors.split(",")],
                abstract=abstract,
                url=link,
                source="arXiv"
            ))
        return papers
