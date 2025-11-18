from abc import ABC, abstractmethod
from typing import List
from data.models import Paper
import json
import logging
import mysql.connector
from datetime import date
from dataclasses import asdict

logging.basicConfig(level=logging.INFO)

class BaseRepository(ABC):
    @abstractmethod
    def save_papers(self, papers: List[Paper]):
        pass

    @abstractmethod
    def load_papers(self) -> List[Paper]:
        pass

class SQLRepository(BaseRepository):
    def __init__(self, host='localhost', user='root', password='      ', database='chatapp'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
        self._connect()

    def _connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            logging.info(f"Đã kết nối tới MySQL server version {self.connection.get_server_info()}")
        except mysql.connector.Error as e:
            logging.error(f"Không thể kết nối database: {e}")
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def save_papers(self, papers: List[Paper]):
        if not self.connection or not self.cursor:
            logging.error("Chưa kết nối database")
            logging.error("Tiến hành connect!")
            self._connect()

        query = """
        INSERT INTO papers (title, authors, abstract, url, source, published_date, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = []
        papers_doesnot_exist = []
        for p in papers:
            if self.paper_exists(paper=p) == False:
                values.append((
                    p.title,
                    json.dumps(p.authors),
                    p.abstract,
                    p.url,
                    p.source,
                    p.published_date,
                    json.dumps(p.embedding)
                ))
                papers_doesnot_exist.append(p)

        try:
            self.cursor.executemany(query, values)
            self.connection.commit()
            logging.info(f"Đã lưu {len(values)} papers")
        except mysql.connector.Error as e:
            logging.error(f"Lỗi khi lưu papers: {e}")
            self.connection.rollback()
        return papers_doesnot_exist
    def load_papers(self) -> List[Paper]:
        if not self.connection or not self.cursor:
            logging.error("Chưa kết nối database")
            logging.error("Tiến hành connect!")
            self._connect()
        papers = []
        try:
            self.cursor.execute("SELECT title, authors, abstract, url, source, published_date, embedding FROM papers")
            rows = self.cursor.fetchall()
            for row in rows:
                papers.append(Paper(
                    title=row[0],
                    authors=json.loads(row[1]),
                    abstract=row[2],
                    url=row[3],
                    source=row[4],
                    published_date=row[5],
                    embedding=json.loads(row[6])
                    
                ))
        except mysql.connector.Error as e:
            logging.error(f"Lỗi khi load papers: {e}")
        return papers

    def paper_exists(self, paper: Paper) -> bool:
        """Kiểm tra xem paper đã tồn tại trong DB chưa dựa trên title"""
        if not self.connection or not self.cursor:
            logging.error("Chưa kết nối database, tiến hành connect lại!")
            self._connect()

        query = "SELECT 1 FROM papers WHERE title=%s LIMIT 1"
        self.cursor.execute(query, (paper.title,))
        return self.cursor.fetchone() is not None

