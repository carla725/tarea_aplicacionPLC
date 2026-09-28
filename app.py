import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.set_page_config(page_title="Resolutor PLC", layout="wide")
st.title("Optimizador de Programación Lineal")

st.markdown("### 1. Definición del Modelo")
col1, col2, col3 = st.columns(3)
with col1:
    tipo_opt = st.selectbox("Objetivo", ["Maximizar", "Minimizar"])
with col2:
    n_vars = st.number_input("Número de variables", min_value=2, max_value=10, value=2)
with col3:
    n_cons = st.number_input("Número de restricciones", min_value=1, max_value=20, value=3)

st.markdown("### 2. Función Objetivo")
cols_obj = st.columns(n_vars)
c = []
for i in range(n_vars):
    with cols_obj[i]:
        coef = st.number_input(f"Coeficiente X{i+1}", value=1.0, key=f"c_{i}")
        c.append(coef)

st.markdown("### 3. Restricciones")
A_ub, b_ub, A_eq, b_eq = [], [], [], []
restricciones_guardadas = []

for i in range(n_cons):
    st.write(f"**Restricción {i+1}**")
    cols_cons = st.columns(n_vars + 2)
    row = []
    for j in range(n_vars):
        with cols_cons[j]:
            row.append(st.number_input(f"X{j+1}", value=1.0, key=f"a_{i}_{j}"))
    
    with cols_cons[n_vars]:
        signo = st.selectbox("Signo", ["<=", ">=", "=="], key=f"signo_{i}")
    
    with cols_cons[n_vars+1]:
        rhs = st.number_input("Lado derecho (RHS)", value=10.0, key=f"rhs_{i}")
        
    restricciones_guardadas.append((row, signo, rhs))

if st.button("Resolver Modelo", type="primary"):
    # Procesar datos para linprog
    c_opt = [-x for x in c] if tipo_opt == "Maximizar" else c
    
    for row, signo, rhs in restricciones_guardadas:
        if signo == "<=":
            A_ub.append(row)
            b_ub.append(rhs)
        elif signo == ">=":
            A_ub.append([-x for x in row])
            b_ub.append(-rhs)
        else:
            A_eq.append(row)
            b_eq.append(rhs)
            
    # Resolver
    bounds = [(0, None) for _ in range(n_vars)] # Variables no negativas
    res = linprog(c_opt, 
                  A_ub=A_ub if A_ub else None, 
                  b_ub=b_ub if b_ub else None,
                  A_eq=A_eq if A_eq else None, 
                  b_eq=b_eq if b_eq else None,
                  bounds=bounds, method='highs')

    st.markdown("---")
    st.markdown("### Resultados de la Solución")
    
    if res.success:
        st.success("¡Solución óptima encontrada!")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.write("**Función Objetivo (Z):**", round(res.fun if tipo_opt == "Minimizar" else -res.fun, 4))
            st.write("**Valor de las Variables:**")
            for i, val in enumerate(res.x):
                st.write(f"X{i+1} = {round(val, 4)}")
                
        with col_res2:
            st.write("**Análisis de Saturación de Restricciones:**")
            for i, (row, signo, rhs) in enumerate(restricciones_guardadas):
                valor_izq = sum(row[j] * res.x[j] for j in range(n_vars))
                holgura = abs(rhs - valor_izq)
                estado = "Saturada (Holgura = 0)" if holgura < 1e-5 else f"No saturada (Holgura = {round(holgura, 4)})"
                st.write(f"R{i+1}: {estado}")
                
        # Método Gráfico (Solo para 2 variables)
        if n_vars == 2:
            st.markdown("### Método Gráfico")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            x_vals = np.linspace(0, max(res.x) * 1.5 + 10, 400)
            
            # Dibujar restricciones
            for i, (row, signo, rhs) in enumerate(restricciones_guardadas):
                if row[1] != 0:
                    y_vals = (rhs - row[0] * x_vals) / row[1]
                    ax.plot(x_vals, y_vals, label=f'R{i+1}')
                else:
                    ax.axvline(x=rhs/row[0], label=f'R{i+1}')
                    
            # Marcar punto óptimo
            ax.plot(res.x[0], res.x[1], 'r*', markersize=15, label='Óptimo')
            
            ax.set_xlim(0, max(res.x) * 1.5 + 5)
            ax.set_ylim(0, max(res.x) * 1.5 + 5)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.axhline(0, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
    else:
        st.error("El modelo no tiene solución factible o es no acotado.")
