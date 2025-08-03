# reports/report_generator.py
def generate_markdown_report(diag_data, filepath="reports/report.md"):
    import os
    os.makedirs("reports", exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:  # ← Added encoding
        f.write("# 🧠 ExplainML++ Report\n\n")
        f.write(f"- Dataset: {diag_data.get('dataset', 'Unknown')}\n")
        f.write(f"- Target: {diag_data['target']}\n")
        f.write(f"- Best Model: {diag_data['best_model']} (F1: {diag_data['f1_score']:.3f})\n")
        
        if diag_data.get("suggestions"):
            f.write("\n## 🛠️ Suggestions\n")
            for s in diag_data["suggestions"]:
                f.write(f"- [{s['priority']}] {s['suggestion']}\n")
    
    print(f"✅ Report saved: {filepath}")

def generate_pdf_report(diag_data, filepath="reports/report.pdf"):
    from fpdf import FPDF
    import os
    os.makedirs("reports", exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    # Use text instead of emoji for PDF
    pdf.cell(0, 10, "ExplainML++ Diagnostic Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Dataset: {diag_data.get('dataset', 'Unknown')}", ln=True)
    pdf.cell(0, 10, f"Target: {diag_data['target']}", ln=True)
    pdf.cell(0, 10, f"Best Model: {diag_data['best_model']}", ln=True)
    pdf.cell(0, 10, f"F1 Score: {diag_data['f1_score']:.3f}", ln=True)

    if diag_data.get("suggestions"):
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Suggestions:", ln=True)
        pdf.set_font("Arial", "", 12)
        for s in diag_data["suggestions"]:
            pdf.cell(0, 10, f"[{s['priority']}] {s['suggestion']}", ln=True)

    pdf.output(filepath)
    print(f"📄 PDF report saved: {filepath}")