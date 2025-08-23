"""
Script de debugging para reportes PDF
"""

import sys
import traceback
import os

# Agregar ruta src
sys.path.append('src')

def test_imports():
    """Probar todas las importaciones necesarias"""
    print("🔍 TESTING IMPORTS...")
    
    try:
        import reportlab
        print(f"✅ ReportLab: {reportlab.Version}")
    except Exception as e:
        print(f"❌ ReportLab error: {e}")
        return False
    
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        print("✅ ReportLab modules imported")
    except Exception as e:
        print(f"❌ ReportLab modules error: {e}")
        return False
    
    try:
        from sistema_multilenguaje.sistema_multilenguaje import t
        print("✅ Sistema multilenguaje imported")
    except Exception as e:
        print(f"⚠️ Sistema multilenguaje error: {e}")
    
    try:
        from reportes_pdf.generador_reportes import GeneradorReportesPDF
        print("✅ GeneradorReportesPDF imported")
    except Exception as e:
        print(f"❌ GeneradorReportesPDF error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    return True

def test_simple_pdf():
    """Crear un PDF simple para probar"""
    print("\n📄 TESTING SIMPLE PDF...")
    
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        import io
        
        # Crear PDF en memoria
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Contenido simple
        story = []
        story.append(Paragraph("TEST PDF - Sistema COVID-19 IA", styles['Title']))
        story.append(Paragraph("Este es un PDF de prueba.", styles['Normal']))
        
        # Construir PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        print(f"✅ PDF simple creado exitosamente ({len(pdf_data)} bytes)")
        
        # Guardar archivo de prueba
        with open("test_pdf.pdf", "wb") as f:
            f.write(pdf_data)
        print("✅ PDF de prueba guardado como 'test_pdf.pdf'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando PDF simple: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_generador_reportes():
    """Probar el generador de reportes específico"""
    print("\n🛠️ TESTING GENERADOR DE REPORTES...")
    
    try:
        from reportes_pdf.generador_reportes import generador_reportes
        
        # Datos de prueba para análisis individual
        resultado_test = {
            'modelo_usado': 'Custom_CNN',
            'clase_predicha': 'Normal',
            'confianza': 85.5,
            'probabilidades': [0.05, 0.10, 0.855, 0.05],
            'archivo_original': 'test_image.jpg',
            'dimensiones_imagen': (256, 256),
            'formato_imagen': 'JPEG'
        }
        
        print("🔄 Probando reporte de análisis individual...")
        pdf_content = generador_reportes.generar_reporte_analisis_individual(resultado_test)
        
        print(f"✅ Reporte generado exitosamente ({len(pdf_content)} bytes)")
        
        # Guardar archivo de prueba
        with open("test_reporte_analisis.pdf", "wb") as f:
            f.write(pdf_content)
        print("✅ Reporte guardado como 'test_reporte_analisis.pdf'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en generador de reportes: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Ejecutar todos los tests"""
    print("🚨 DEBUGGING SISTEMA DE REPORTES PDF")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ FALLÓ EN IMPORTS - Revisar instalaciones")
        return
    
    # Test 2: PDF Simple
    if not test_simple_pdf():
        print("\n❌ FALLÓ EN PDF SIMPLE - Problema con ReportLab")
        return
    
    # Test 3: Generador específico
    if not test_generador_reportes():
        print("\n❌ FALLÓ EN GENERADOR - Revisar código")
        return
    
    print("\n" + "=" * 50)
    print("🎉 TODOS LOS TESTS PASARON - SISTEMA PDF FUNCIONAL")
    print("💡 Si aún tienes problemas, es un issue de Streamlit")

if __name__ == "__main__":
    main()