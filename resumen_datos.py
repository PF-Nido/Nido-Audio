import os
from collections import defaultdict
import pandas as pd

def resumir_directorio(directorio, file_out):
    if not os.path.exists(directorio):
        return False
    
    file_out.write(f'\n' + '='*50 + '\n')
    file_out.write(f'Resumen de: {directorio}\n')
    file_out.write('='*50 + '\n')
    
    resumen = defaultdict(lambda: {'audios_json': 0, 'archivos_embeddings': 0, 'segmentos_embeddings': 0})
    
    # Comprobar si hay un archivo consolidado en vez de archivos individuales
    _path_limpios_parquet = os.path.join(directorio, 'embeddings_limpios.parquet')
    if os.path.exists(_path_limpios_parquet):
        try:
            df = pd.read_parquet(_path_limpios_parquet)
            if 'genero' in df.columns and 'especie' in df.columns and 'xc_id' in df.columns:
                df['nombre_cientifico'] = df['genero'] + ' ' + df['especie']
                for especie, group in df.groupby('nombre_cientifico'):
                    resumen[especie]['audios_json'] = group['xc_id'].nunique()
                    resumen[especie]['archivos_embeddings'] = 1
                    resumen[especie]['segmentos_embeddings'] = len(group)
        except Exception as e:
            file_out.write(f'Error leyendo parquet consolidado: {e}\n')
    else:
        try:
            archivos = os.listdir(directorio)
        except Exception:
            return False
            
        for archivo in archivos:
            if archivo.endswith('.json') and not archivo.startswith('eBird'):
                partes = archivo.replace('.json', '').split('_', 1)
                if len(partes) > 1:
                    especie = partes[1].replace('_', ' ')
                    resumen[especie]['audios_json'] += 1
            elif archivo.endswith('.parquet') or archivo.endswith('.csv'):
                if 'consolidado' in archivo: continue
                nombre_base = archivo.replace('_embeddings.parquet', '').replace('_embeddings.csv', '')
                partes = nombre_base.split('_', 1)
                if len(partes) > 1:
                    especie = partes[1].replace('_', ' ')
                    resumen[especie]['archivos_embeddings'] += 1
                    try:
                        ruta = os.path.join(directorio, archivo)
                        df = pd.read_parquet(ruta) if archivo.endswith('.parquet') else pd.read_csv(ruta)
                        resumen[especie]['segmentos_embeddings'] += len(df)
                    except:
                        pass
                        
    if not resumen:
        file_out.write('  Carpeta vacía o sin datos reconocidos.\n\n')
        return True
        
    df_resumen = pd.DataFrame.from_dict(resumen, orient='index')
    df_resumen.index.name = 'Especie'
    df_resumen = df_resumen.reset_index().sort_values('Especie')
    
    file_out.write(df_resumen.to_string(index=False) + '\n\n')
    file_out.write('[TOTALES]\n')
    file_out.write(f'Total especies:                {len(df_resumen)}\n')
    file_out.write(f"Total audios:                  {df_resumen['audios_json'].sum()}\n")
    file_out.write(f"Total archivos embeddings:     {df_resumen['archivos_embeddings'].sum()}\n")
    file_out.write(f"Total segmentos (embeddings):  {df_resumen['segmentos_embeddings'].sum()}\n\n")
    
    print(f'Procesado {directorio}.')
    return True

if __name__ == '__main__':
    carpetas = [
        'analisis_datos/datos_fase1',
        'limpieza_datos/datos_fase2',
        'limpieza_datos/datos_fase2v2'
    ]
    
    encontradas = 0
    with open('reporte_resumen.txt', 'w', encoding='utf-8') as f:
        for c in carpetas:
            if resumir_directorio(c, f):
                encontradas += 1
                
    if encontradas == 0:
        print('No se encontró ninguna de las carpetas especificadas.')
    else:
        print('Resumen generado exitosamente en el archivo reporte_resumen.txt')

