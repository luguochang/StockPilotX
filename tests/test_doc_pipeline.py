from __future__ import annotations

import unittest

from backend.app.data.ingestion import IngestionService, IngestionStore
from backend.app.data.sources import AnnouncementService, QuoteService, TencentAdapter


class DocumentPipelineTestCase(unittest.TestCase):
    """RAG-003：文档处理流水线与复核队列测试。"""

    def setUp(self) -> None:
        self.store = IngestionStore()
        self.svc = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=AnnouncementService(adapters=[]),
            store=self.store,
        )

    def test_html_doc_index_extracts_tables(self) -> None:
        html = "<html><body><h1>财报</h1><p>A|B|C</p><p>1|2|3</p></body></html>"
        up = self.svc.upload_doc("doc-html", "report.html", html, "upload")
        idx = self.svc.index_doc("doc-html")
        self.assertEqual(up["status"], "uploaded")
        self.assertEqual(idx["status"], "indexed")
        self.assertGreaterEqual(idx["table_count"], 1)
        self.assertGreater(idx["chunk_count"], 0)

    def test_pdf_low_confidence_goes_review_queue(self) -> None:
        self.svc.upload_doc("doc-pdf", "report.pdf", "PDF文本内容", "upload")
        self.assertGreaterEqual(len(self.store.review_queue), 1)
        self.assertEqual(self.store.review_queue[0]["doc_id"], "doc-pdf")


if __name__ == "__main__":
    unittest.main()

