# Generates a tiny sample PDF for quick testing.
from fpdf import FPDF
import os
os.makedirs('data/pdfs', exist_ok=True)
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', size=12)
pdf.multi_cell(0, 6, 'Sample Document Title\n\nThis is a sample PDF used to test the RAG ingestion pipeline.\n\nRefund policy:\nIf you request a refund within 30 days of purchase, you will be refunded in full.')
pdf.output('data/pdfs/sample.pdf')
print('Wrote data/pdfs/sample.pdf')
