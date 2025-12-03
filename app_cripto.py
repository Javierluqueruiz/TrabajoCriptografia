import streamlit as st
import numpy as np
import math

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Demo Criptografía Unimodular", layout="wide")

st.title("🔐 Criptografía Unimodular: Laboratorio Interactivo")
st.markdown("Generación de claves, cifrado y análisis de errores basado en Koshkin & Styers (2017).")

# --- BARRA LATERAL: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Configuración de Claves")
    
    st.subheader("Matriz Unimodular U")
    col1, col2 = st.columns(2)
    u11 = col1.number_input("u11", value=3, step=1)
    u12 = col2.number_input("u12", value=1, step=1)
    u21 = col1.number_input("u21", value=2, step=1)
    u22 = col2.number_input("u22", value=1, step=1)
    
    st.subheader("Vector Inicial V0")
    col3, col4 = st.columns(2)
    a0 = col3.number_input("A0", value=1, step=1)
    b0 = col4.number_input("B0", value=0, step=1)
    
    st.subheader("Clave Privada n")
    n = st.slider("Valor de n", min_value=1, max_value=20, value=3)

# --- LÓGICA MATEMÁTICA ---
# 1. Cálculos base
t = u11 + u22  # Traza
d = (u11 * u22) - (u12 * u21) # Determinante U

# 2. Construcción Semilla M0 consistente
# Primera columna = U * V0
a1 = u11 * a0 + u12 * b0
b1 = u21 * a0 + u22 * b0
M0 = np.array([[a1, a0], [b1, b0]])
mu = int(a1 * b0) - int(a0 * b1)  # Determinante M0

# 3. Razón Unimodular
try:
    phi = (t + math.sqrt(t**2 - 4*d)) / 2
except ValueError:
    phi = 0 # Manejo de errores si d es muy grande negativo

# 4. Generación Mn (Recurrencia)
def generar_secuencia(start_vals, n, t, d):
    seq = list(start_vals) # [Val_0, Val_1]
    for i in range(2, n + 2):
        val_next = t * seq[-1] - d * seq[-2]
        seq.append(val_next)
    return seq

seq_A = generar_secuencia([a0, a1], n, t, d)
seq_B = generar_secuencia([b0, b1], n, t, d)

# Mn se forma con los términos n+1 y n (índices invertidos respecto a visualización de lista)
Mn = np.array([
    [seq_A[n+1], seq_A[n]],
    [seq_B[n+1], seq_B[n]]
])


# --- INTERFAZ PRINCIPAL ---

# BLOQUE 1: ANÁLISIS DEL SISTEMA
st.header("2. Análisis del Sistema Generado")
col_sys1, col_sys2, col_sys3, col_sys4 = st.columns(4)

with col_sys1:
    st.info(f"**Invariantes:**\n\nTraza (t) = {t}\n\nDet (d) = {d}")
    if abs(d) != 1:
        st.error("⚠️ ¡La matriz U no es unimodular (det != ±1)!")

with col_sys2:
    st.success(f"**Razón Unimodular ($\phi$):**\n\n {phi:.5f}")

det_mn_real = int(Mn[0,0]) * int(Mn[1,1]) - int(Mn[0,1]) * int(Mn[1,0])

with col_sys3: 
    st.latex(r"M_0 = \begin{pmatrix} " + f"{M0[0,0]} & {M0[0,1]} \\\\ {M0[1,0]} & {M0[1,1]}" + r" \end{pmatrix}")
    with st.expander("Determinante"):
        st.latex(f"\u03bc =  {mu}")

with col_sys4:
    st.latex(r"M_n = \begin{pmatrix} " + f"{Mn[0,0]} & {Mn[0,1]} \\\\ {Mn[1,0]} & {Mn[1,1]}" + r" \end{pmatrix}")
    st.caption(f"Determinante de Mn: {det_mn_real} (Esperado: {(d**n) * mu})")
    with st.expander("Determinante esperado"):
        st.latex(f"Det(M_n) = d^n * \u03bc = ({d})^{n} * {mu} = {(d**n) * mu}")

st.divider()

# BLOQUE 2: CIFRADO DE TEXTO
st.header("3. Cifrado de Mensaje")

texto = st.text_input("Introduce un mensaje de 4 letras:", value="HOLA", max_chars=4).upper()

# Convertir texto a números (A=0, B=1...)
alfabeto = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
if len(texto) < 4:
    texto = texto.ljust(4, 'X') # Relleno

vals = [alfabeto.index(c) if c in alfabeto else 0 for c in texto]
P = np.array([[vals[0], vals[1]], [vals[2], vals[3]]])
det_P = int(P[0,0]) * int(P[1,1]) - int(P[0,1]) * int(P[1,0])

# Cifrar C = P * Mn
C = np.dot(P, Mn)

col_cif1, col_cif2 = st.columns(2)
with col_cif1:
    st.markdown("**Matriz Texto Claro (P):**")
    st.write(P)
    st.metric("Check Number (Det P)", det_P)

with col_cif2:
    st.markdown("**Matriz Cifrada (C=P*Mn):**")
    st.write(C)
    st.latex(r"C=\begin{pmatrix} " + f"{C[0,0]} & {C[0,1]} \\\\ {C[1,0]} & {C[1,1]}" + r" \end{pmatrix}")
    
st.divider()

# BLOQUE 3: SIMULACIÓN DE CANAL RUIDOSO
st.header("4. Simulación de Errores y Verificación")
st.markdown("Modifica los valores recibidos para simular ruido en el canal.")

# Inputs para modificar la matriz recibida
col_err1, col_err2 = st.columns(2)
c11_rx = col_err1.number_input("C11 (Recibido)", value=int(C[0,0]))
c12_rx = col_err2.number_input("C12 (Recibido)", value=int(C[0,1]))
c21_rx = col_err1.number_input("C21 (Recibido)", value=int(C[1,0]))
c22_rx = col_err2.number_input("C22 (Recibido)", value=int(C[1,1]))

C_rx = np.array([[c11_rx, c12_rx], [c21_rx, c22_rx]])
det_rx = int(C_rx[0,0]) * int(C_rx[1,1]) - int(C_rx[0,1]) * int(C_rx[1,0])
det_esperado = mu * (d**n) * det_P

st.latex(f"Det(C) = {det_rx}")

# --- ANÁLISIS DE ERRORES ---
st.subheader("Diagnóstico del Receptor")

# 1. Determinante
match_det = (det_rx == det_esperado)
st.write(f"📡 **Verificación de Determinante:**")
if match_det:
    st.success(f"✅ COINCIDE. Det Recibido ({det_rx}) == Esperado ({det_esperado})")
else:
    st.error(f"❌ ERROR DETECTADO. Det Recibido ({det_rx}) != Esperado ({det_esperado})")
with st.expander("Cálculo"):
    st.markdown(f"Determinante Esperado:") 
    st.latex(f"\u03bc * d^n * Det(P) = {mu} * ({d})^{n} * {det_P} = {det_esperado}")



# 2. Ratios Unimodulares
st.write("📐 **Análisis de Ratios (Filas):**")
r1 = C_rx[0,0] / C_rx[0,1] if C_rx[0,1] != 0 else 0
r2 = C_rx[1,0] / C_rx[1,1] if C_rx[1,1] != 0 else 0

col_rat1, col_rat2 = st.columns(2)
with col_rat1:
    delta1 = abs(r1 - phi)
    estado1 = "✅ OK" if delta1 < 0.1 else "⚠️ SOSPECHOSO"
    st.metric("Ratio Fila 1", f"{r1:.4f}", delta=f"Desv: {delta1:.4f}", delta_color="inverse")
    st.caption(estado1)

with col_rat2:
    delta2 = abs(r2 - phi)
    estado2 = "✅ OK" if delta2 < 0.1 else "⚠️ SOSPECHOSO"
    st.metric("Ratio Fila 2", f"{r2:.4f}", delta=f"Desv: {delta2:.4f}", delta_color="inverse")
    st.caption(estado2)

# 3. Ratio de Columna (La novedad del paper)
st.write("📊 **Ratio de Columna (Para Errores Dobles):**")
if C_rx[0,0] != 0 and C_rx[0,1] != 0:
    col_ratio1 = C_rx[1,0] / C_rx[0,0]
    col_ratio2 = C_rx[1,1] / C_rx[0,1]
    st.info(f"Columna 1: {col_ratio1:.4f} | Columna 2: {col_ratio2:.4f}")
    if abs(col_ratio1 - col_ratio2) < 0.01:
        st.markdown("**Conclusión:** Las columnas son consistentes entre sí.")
    else:
        st.markdown("**Conclusión:** Inconsistencia entre columnas detectada.")
else:
    st.warning("No se puede calcular (división por cero).")
    

# ... (Pega esto al final de tu archivo app_cripto.py) ...

st.divider()

# BLOQUE 5: RECUPERACIÓN AUTOMÁTICA
st.header("5. Recuperación y Corrección de Errores")
st.markdown("""
Si la verificación falla, el receptor utiliza el **Determinante Esperado** y el **Ratio de Columna** (enviado como dato de control) para reconstruir matemáticamente los datos perdidos.
""")

# 1. Simulamos que recibimos el Ratio de Columna correcto (del emisor)
# Nota: En un caso real, esto se envía junto con det_P.
if C[0,0] != 0:
    sent_col_ratio = C[1,0] / C[0,0]
else:
    sent_col_ratio = 0
    
st.info(f"ℹ️ **Dato de Control Adicional:** Ratio de Columna Real = {sent_col_ratio:.5f}")

if st.button("🛠️ Intentar Recuperar Matriz y Mensaje"):
    
    # Función de recuperación robusta (Fuerza bruta guiada)
    def recover_matrix(Bad_C, target_det, col_ratio, phi):
        candidates = []
        search_range = 2000 # Rango de búsqueda alrededor de la estimación
        best_solution = None
        min_dist = float('inf')
        
        # --- ESTRATEGIA A: Asumir Error en Fila 1 (Fila 2 es correcta) ---
        # Incógnitas: x(c11), y(c12). Conocidos: c21, c22
        c21, c22 = int(Bad_C[1,0]), int(Bad_C[1,1])
        
        # Estimamos x usando el ratio de columna: x = c21 / ratio
        if c21 != 0 and col_ratio != 0:
            # 1. ¿Dónde DEBERÍA estar x según el ratio de columna?
            est_x = c21 / col_ratio
            
            # 2. Buscamos alrededor de esa estimación
            start_x = int(est_x) - search_range
            end_x = int(est_x) + search_range
            
            for x in range(start_x, end_x):
                # Despejamos y: y = (x*c22 - det) / c21
                numerator = x * c22 - target_det
                
                if numerator % c21 == 0: # Si la división es exacta
                    y = numerator // c21
                    
                    # VALIDACIÓN:
                    # Calculamos cuán lejos está este candidato de la predicción del ratio de columna
                    # Distancia = |x_candidato - x_estimado|
                    dist = abs(x - est_x)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_solution = np.array([[x, y], [c21, c22]])

        # --- ESTRATEGIA B: Asumir Error en Fila 2 (Fila 1 es correcta) ---
        # Incógnitas: x(c21), y(c22). Conocidos: c11, c12
        c11, c12 = int(Bad_C[0,0]), int(Bad_C[0,1])
        
        if c11 != 0:
            # 1. Predicción: x = c11 * col_ratio
            est_x = c11 * col_ratio
            
            start_x = int(est_x) - search_range
            end_x = int(est_x) + search_range
            
            for x in range(start_x, end_x):
                # Ecuación: c11*y - c12*x = det  ->  y = (det + c12*x) / c11
                numerator = target_det + c12 * x
                
                if numerator % c11 == 0:
                    y = numerator // c11
                    # Distancia a la predicción
                    dist = abs(x - est_x)
                    
                    # ¿Es este candidato mejor que lo que encontramos en la Estrategia A?
                    if dist < min_dist:
                        min_dist = dist
                        best_solution = np.array([[c11, c12], [x, y]])

        return best_solution

    # Lógica Principal
    if det_rx == det_esperado:
        st.success("✅ La matriz recibida es correcta. No hace falta recuperar nada.")
        C_final = C_rx
    else:
        with st.spinner("Calculando posibles soluciones..."):
            C_final = recover_matrix(C_rx, det_esperado, sent_col_ratio, phi)
        
        if C_final is not None:
            st.success("✅ ¡Matriz Reconstruida con Éxito!")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("Tu Matriz con Ruido:", C_rx)
            with col_res2:
                st.write("Matriz Recuperada:", C_final)
        else:
            st.error("❌ No se pudo recuperar. El daño es demasiado complejo o el rango de búsqueda insuficiente.")

    # Descifrado Final (Volver a texto)
    if C_final is not None:
        st.markdown("### Descifrado del Mensaje")
        # P = C * Mn^-1
        # Inversa manual para mantener enteros: Inv = 1/det * Adjunta
        det_mn = int(Mn[0,0]*Mn[1,1] - Mn[0,1]*Mn[1,0])
        
        if det_mn != 0:
            # Matriz Adjunta de Mn
            adj_Mn = np.array([[Mn[1,1], -Mn[0,1]], [-Mn[1,0], Mn[0,0]]])
            
            # P_temp = C * Adjunta
            P_temp = np.dot(C_final, adj_Mn)
            
            # Dividimos por el determinante (debe dar exacto si todo fue bien)
            P_rec = P_temp // det_mn
            
            st.write("Matriz P Original Restaurada:", P_rec)
            
            # Convertir números a letras
            msg_rec = ""
            for val in P_rec.flatten():
                idx = int(val)
                if 0 <= idx < len(alfabeto):
                    msg_rec += alfabeto[idx]
                else:
                    msg_rec += "?"
            
            st.title(f"📜 Mensaje: {msg_rec}")
        else:
            st.error("La matriz clave Mn es singular (det=0), no se puede descifrar.")