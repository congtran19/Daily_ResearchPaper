from data.repositories.base_repository import BaseRepository
from data.sources.arxiv_source import ArxivSource
import logging
class FetchPapersPipeline:
    def __init__(self, repo: BaseRepository, telegram_token: str = None, chat_id: str = None):
        self.source = ArxivSource()  # Có thể mở rộng với Factory cho nhiều sources
        self.repo = repo
        # self.translator = TranslatorService()
        # self.telegram = TelegramService(telegram_token, chat_id) if telegram_token and chat_id else None

    def run(self, limit: int = 5):
        """Run pipeline: Fetch papers, add to JSON, translate titles, notify via Telegram."""
        new_papers = self.source.fetch_latest(limit)
        if new_papers:
            # Add to existing papers (dùng add_papers để tránh overwrite duplicate)
            paper_doestnotexist = self.repo.save_papers(new_papers)
            logging.warning(len(paper_doestnotexist))
            print(f"Added new papers to database.")
            return paper_doestnotexist
            # for paper in new_papers:
            #     vi_title = self.translator.translate(paper.title, target_lang="vi")
            #     message = f"📄 {vi_title}\nSource: {paper.source}\nURL: {paper.url}"
            #     if self.telegram:
            #         self.telegram.send_message(message)
            #     else:
            #         print(f"Mock Telegram send: {message}")