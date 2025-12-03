###################
## IMPORTACIONES ##
###################

import random
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Ataque de Texto Elegido",
    layout="wide",
    initial_sidebar_state="collapsed"
)

##########################################
## GENERAR CLAVE PRIVADA ALEATORIAMENTE ##
##########################################

if 'n' not in st.session_state:
    st.session_state.n = random.randint(5,15)

n = st.session_state.n


#######################
## LÓGICA DE CIFRADO ##
#######################

def calcular_matriz_cifrada(matriz_P, n):
    Q = np.array([[1, 1], 
                  [1, 0]])

    Q_n = np.linalg.matrix_power(Q, n)

    matriz_C = matriz_P @ Q_n
    
    return matriz_C


#########################
## INTERFAZ DE USUARIO ## 
#########################

def ventana_principal():
    st.markdown("<h1 style='text-align: center;'>Simulador de Ataque de Texto Elegido</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Sistema de Criptografía Áurea</h3>", unsafe_allow_html=True)
    st.write("") 
    
    tab1, tab2, tab3 = st.tabs(["Cifrar mensaje", "Adivinar clave", "Sucesión de Fibonacci"])


    with tab1:
        st.write("Introduce valores numéricos enteros para la matriz $P$:")
        
        col_main_1, col_main_2 = st.columns([1, 2])
        
        with col_main_1:
            st.caption("Entrada (Matriz 2x2 de enteros)")
            c1, c2 = st.columns(2)
            
            with c1:
                p00 = st.text_input("Fila 1, Col 1", value="0", key="p00")
                p10 = st.text_input("Fila 2, Col 1", value="0", key="p10")
            with c2:
                p01 = st.text_input("Fila 1, Col 2", value="0", key="p01")
                p11 = st.text_input("Fila 2, Col 2", value="0", key="p11")

            st.write("")
            btn_cifrar = st.button("Cifrar matriz", type="primary", use_container_width=True)

        with col_main_2:
            if btn_cifrar:
                try:
                    datos_lista = [
                        [int(p00), int(p01)], 
                        [int(p10), int(p11)]
                    ]
                    mi_array_numpy = np.array(datos_lista)

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

    with tab2:
        st.write("Intenta deducir la clave privada $n$.")
        
        col_clave, _ = st.columns([1, 2])
        
        with col_clave:
            valor_input = st.text_input("Escribe el valor de la clave privada:", key="input_clave")
            
            if st.button("Comprobar clave", use_container_width=True):
                if valor_input is not None:
                    if valor_input.strip() == "" or not valor_input.isdigit():
                        st.warning("Entrada inválida. Debe ser un número entero positivo.")
                    elif int(valor_input) == n:
                        st.balloons()
                        st.success(f"¡Correcto! Has adivinado la clave privada: {valor_input}")
                    else:
                        st.error(f"¡Incorrecto! La clave privada no es {valor_input}.")

    with tab3:
        st.subheader("Primeros 15 números")
        
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


##########
## MAIN ##
##########

if __name__ == "__main__":
    ventana_principal()