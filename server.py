"""
Servidor HTTP simple para servir el dashboard EMG
Ejecuta en localhost:8013
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8013

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Agregar headers CORS para permitir carga de recursos
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Personalizar mensajes de log
        print(f"[{self.log_date_time_string()}] {format % args}")

def main():
    # Cambiar al directorio del script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Verificar que los archivos necesarios existan
    dashboards = []
    if Path('dashboard.html').exists():
        dashboards.append(('dashboard.html', 'Dashboard Original'))
    if Path('dashboard_v2.html').exists():
        dashboards.append(('dashboard_v2.html', 'Dashboard v2 (Con Pestañas)'))
    
    if not dashboards:
        print("ERROR: No se encontró ningún dashboard.html")
        return
    
    if not Path('data.json').exists():
        print("ERROR: data.json no encontrado. Ejecuta process_emg.py primero.")
        return
    
    # Crear servidor
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 60)
        print(f"Servidor EMG Dashboard iniciado")
        print(f"Puerto: {PORT}")
        print("=" * 60)
        print("\nDashboards disponibles:")
        for filename, description in dashboards:
            url = f"http://localhost:{PORT}/{filename}"
            print(f"  - {description}: {url}")
        print("\nPresiona Ctrl+C para detener el servidor\n")
        
        # Abrir primer dashboard automáticamente
        try:
            webbrowser.open(f"http://localhost:{PORT}/{dashboards[0][0]}")
        except:
            print(f"No se pudo abrir el navegador automáticamente.")
        
        # Iniciar servidor
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServidor detenido por el usuario.")
            httpd.shutdown()

if __name__ == '__main__':
    main()

