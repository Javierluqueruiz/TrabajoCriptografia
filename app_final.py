import streamlit as st
import numpy as np
import math
import random

# ==============================================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Estudio Comparativo: Criptografía Dorada vs Unimodular",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔐 Estudio Comparativo de Criptografía")
st.sidebar.header("Navegación")
sistema = st.sidebar.radio(
    "Selecciona el sistema a estudiar:",
    ["🏆 Criptografía Áurea", "🛡️ Criptografía Unimodular"]
)

st.sidebar.markdown("---")

# ==============================================================================
# PESTAÑA 1: CRIPTOGRAFÍA ÁUREA (Código de tu compañero)
# ==============================================================================
if sistema == "🏆 Criptografía Áurea":
    
    st.header("1. Simulador de Ataque de Texto Elegido")
    st.markdown("<h4 style='color: gray;'>Sistema de Criptografía Áurea</h4>", unsafe_allow_html=True)

    # --- LÓGICA DE ESTADO (SESSION STATE) ---
    if 'n' not in st.session_state:
        st.session_state.n = random.randint(5, 15)
    
    n = st.session_state.n

    # --- FUNCIÓN DE CIFRADO GOLDEN ---
    def calcular_matriz_cifrada(matriz_P, n):
        Q = np.array([[1, 1], 
                  [1, 0]])
        Q_n = np.linalg.matrix_power(Q, n)
        matriz_C = matriz_P @ Q_n
        return matriz_C

    # --- SUB-PESTAÑAS INTERNAS ---
    subtab1, subtab2, subtab3 = st.tabs(["Cifrar mensaje", "Adivinar clave", "Sucesión de Fibonacci"])

    with subtab1:
        st.write("Introduce valores numéricos enteros para la matriz $P$:")
        col_main_1, col_main_2 = st.columns([1, 2])
        
        with col_main_1:
            st.caption("Entrada (Matriz 2x2 de enteros)")
            c1, c2 = st.columns(2)
            with c1:
                p00 = st.text_input("Fila 1, Col 1", value="0", key="g_p00")
                p10 = st.text_input("Fila 2, Col 1", value="0", key="g_p10")
            with c2:
                p01 = st.text_input("Fila 1, Col 2", value="0", key="g_p01")
                p11 = st.text_input("Fila 2, Col 2", value="0", key="g_p11")
            
            st.write("")
            btn_cifrar = st.button("Cifrar matriz (Golden)", type="primary", use_container_width=True)

        with col_main_2:
            if btn_cifrar:
                try:
                    datos_lista = [[int(p00), int(p01)], [int(p10), int(p11)]]
                    mi_array_numpy = np.array(datos_lista, dtype=object)
                    matriz_cifrada = calcular_matriz_cifrada(mi_array_numpy, n)
                    
                    st.success("¡Cifrado completado con éxito!")
                    
                    c00, c01 = matriz_cifrada[0][0], matriz_cifrada[0][1]
                    c10, c11 = matriz_cifrada[1][0], matriz_cifrada[1][1]

                    st.latex(r'''
                    C = P \times Q^n = 
                    \begin{pmatrix} 
                    %d & %d \\ 
                    %d & %d 
                    \end{pmatrix}
                    ''' % (c00, c01, c10, c11))
                except ValueError:
                    st.error("Error: entrada inválida. Introduce solo números enteros.")

    with subtab2:
        st.write("Intenta deducir la clave privada $n$.")
        col_clave, _ = st.columns([1, 2])
        with col_clave:
            valor_input = st.text_input("Escribe el valor de la clave privada:", key="input_clave_golden")
            if st.button("Comprobar clave", use_container_width=True):
                if valor_input and valor_input.strip().isdigit():
                    val_int = int(valor_input)
                    if val_int == n:
                        st.balloons()
                        st.success(f"¡Correcto! Has adivinado la clave privada: {val_int}")
                    else:
                        st.error(f"¡Incorrecto! La clave privada no es {val_int}.")
                else:
                    st.warning("Entrada inválida.")

    with subtab3:
        st.subheader("Primeros 15 números de Fibonacci")
        a, b = 1, 1
        serie = [a, b]
        for _ in range(13):
            a, b = b, a + b
            serie.append(b)
        
        cols = st.columns(5)
        for i, numero in enumerate(serie):
            col_index = i % 5
            with cols[col_index]:
                with st.container(border=True):
                    st.markdown(f"<div style='text-align: center; font-weight: bold;'>#{i+1}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 20px;'>{numero}</div>", unsafe_allow_html=True)


# ==============================================================================
# PESTAÑA 2: CRIPTOGRAFÍA UNIMODULAR (Tu código completo)
# ==============================================================================
elif sistema == "🛡️ Criptografía Unimodular":
    st.header("2. Criptografía Unimodular: Laboratorio Interactivo")
    st.markdown("Generación de claves, cifrado y recuperación de errores basado en Koshkin & Styers (2017).")

    with st.sidebar:
        st.header("2.1. Configuración de Claves")
        
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
    st.header("2.2. Análisis del Sistema Generado")
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
    st.header("2.3. Cifrado de Mensaje")

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
    st.header("2.4. Simulación de Errores y Verificación")
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
    st.header("2.5. Recuperación y Corrección de Errores")
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
            search_range = 3000 # Rango de fuerza bruta guiada
            
            # Helper para calcular "puntuación" de una solución candidata
            # Cuanto menor sea el score, más se parece a una matriz unimodular válida
            def calculate_score(M):
                # Error respecto a Phi (filas)
                r1 = M[0,0]/M[0,1] if M[0,1]!=0 else 0
                r2 = M[1,0]/M[1,1] if M[1,1]!=0 else 0
                err_phi = abs(r1 - phi) + abs(r2 - phi)
                
                # Error respecto al Ratio de Columna (si es confiable)
                if col_ratio != 0 and M[0,0] != 0:
                    rc = M[1,0] / M[0,0]
                    err_col = abs(rc - col_ratio)
                else:
                    err_col = 0
                return err_phi + err_col

            # ==============================================================================
            # ESTRATEGIA 1: ERRORES DE FILA (Row Errors)
            # Usamos el Ratio de Columna para estimar x
            # ==============================================================================
            
            # Caso A: Fila 1 Mala (x, y) - Fila 2 Buena (c21, c22)
            c21, c22 = int(Bad_C[1,0]), int(Bad_C[1,1])
            if c21 != 0 and col_ratio != 0:
                est_x = c21 / col_ratio # Predicción
                for x in range(int(est_x)-search_range, int(est_x)+search_range):
                    num = x * c22 - target_det
                    if num % c21 == 0:
                        y = num // c21
                        candidates.append(np.array([[x, y], [c21, c22]], dtype=object))

            # Caso B: Fila 2 Mala (x, y) - Fila 1 Buena (c11, c12)
            c11, c12 = int(Bad_C[0,0]), int(Bad_C[0,1])
            if c11 != 0: # Aquí usamos col_ratio inverso o predicción directa
                est_x = c11 * col_ratio
                for x in range(int(est_x)-search_range, int(est_x)+search_range):
                    num = target_det + c12 * x
                    if num % c11 == 0: # Typo fix: num, no numerator
                        y = num // c11
                        candidates.append(np.array([[c11, c12], [x, y]], dtype=object))

            # ==============================================================================
            # ESTRATEGIA 2: ERRORES DE COLUMNA (Column Errors)
            # Aquí NO sirve el ratio de columna. Usamos Phi para estimar.
            # Ref: Koshkin & Styers, Sección 5, Eq para Column Errors
            # ==============================================================================

            # Caso C: Columna 1 Mala (x, z) - Columna 2 Buena (c12, c22)
            # Incógnitas: x=c11, z=c21. 
            # Estimación: x ~ c12 * phi
            c12, c22 = int(Bad_C[0,1]), int(Bad_C[1,1])
            if c12 != 0:
                est_x = c12 * phi
                for x in range(int(est_x)-search_range, int(est_x)+search_range):
                    # Ecuación: x*c22 - c12*z = det  =>  z = (x*c22 - det) / c12
                    num = x * c22 - target_det
                    if num % c12 == 0:
                        z = num // c12
                        candidates.append(np.array([[x, c12], [z, c22]], dtype=object))

            # Caso D: Columna 2 Mala (y, v) - Columna 1 Buena (c11, c21)
            # Incógnitas: y=c12, v=c22.
            # Estimación: y ~ c11 / phi
            c11, c21 = int(Bad_C[0,0]), int(Bad_C[1,0])
            if phi != 0:
                est_y = c11 / phi
                for y in range(int(est_y)-search_range, int(est_y)+search_range):
                    # Ecuación: c11*v - y*c21 = det => v = (det + y*c21) / c11
                    if c11 != 0:
                        num = target_det + y * c21
                        if num % c11 == 0:
                            v = num // c11
                            candidates.append(np.array([[c11, y], [c21, v]], dtype=object))

            # ==============================================================================
            # ESTRATEGIA 3: ERRORES DIAGONALES (Diagonal Errors)
            # Problema de Factorización guiada por Phi
            # Ref: Koshkin & Styers, Sección 5, Diagonal Errors
            # ==============================================================================

            # Caso E: Diagonal Principal Mala (x, v) - Anti-diag Buena (c12, c21)
            # Ecuación: x*v = det + c12*c21 = K
            # Estimación: x ~ c12 * phi
            c12, c21 = int(Bad_C[0,1]), int(Bad_C[1,0])
            K = target_det + c12 * c21
            est_x = c12 * phi
            
            # Iteramos buscando factores de K cercanos a est_x
            # Nota: K puede ser negativo, cuidado con el rango
            start_x = int(est_x) - search_range
            end_x = int(est_x) + search_range
            
            # Evitamos 0 para no dividir
            if start_x <= 0 <= end_x: 
                rango = list(range(start_x, 0)) + list(range(1, end_x))
            else:
                rango = range(start_x, end_x)

            for x in rango:
                if x != 0 and K % x == 0:
                    v = K // x
                    candidates.append(np.array([[x, c12], [c21, v]], dtype=object))

            # Caso F: Anti-Diagonal Mala (y, z) - Diag Principal Buena (c11, c22)
            # Ecuación: y*z = c11*c22 - det = K
            # Estimación: y ~ c11 / phi
            c11, c22 = int(Bad_C[0,0]), int(Bad_C[1,1])
            K = c11 * c22 - target_det
            if phi != 0:
                est_y = c11 / phi
                start_y = int(est_y) - search_range
                end_y = int(est_y) + search_range
                
                # Manejo de rango seguro
                if start_y <= 0 <= end_y:
                    rango = list(range(start_y, 0)) + list(range(1, end_y))
                else:
                    rango = range(start_y, end_y)

                for y in rango:
                    if y != 0 and K % y == 0:
                        z = K // y
                        candidates.append(np.array([[c11, y], [z, c22]], dtype=object))

            # ==============================================================================
            # SELECCIÓN DEL MEJOR CANDIDATO
            # ==============================================================================
            best_M = None
            min_score = float('inf')
            
            for M in candidates:
                score = calculate_score(M)
                if score < min_score:
                    min_score = score
                    best_M = M
                    
            return best_M
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