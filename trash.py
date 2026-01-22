def compute_eig_smallest_nonzero(L: np.ndarray, kernel_dim: int) -> float:
    """
    Вычисляет наименьшее ненулевое собственное значение матрицы Лапласа.
    
    Для матрицы Лапласа графа:
    - Первое собственное значение всегда 0 (соотв. постоянному вектору)
    - Второе наименьшее значение называется algebraic connectivity
    - kernel_dim = 1 для связных графов (ядро размерности 1)
    
    Parameters
    ----------
    L : np.ndarray
        Матрица Лапласа (симметричная, положительно полуопределённая)
    kernel_dim : int
        Размерность ядра (количество нулевых собственных значений)
        Для связного графа = 1
        
    Returns
    -------
    float
        Наименьшее ненулевое собственное значение
        Или 0.0, если значение меньше порога 1e-12
        
    Notes
    -----
    - Использует алгоритм Ланцоша (eigsh) для поиска наименьших значений
    - Ищет kernel_dim+2 значений, чтобы гарантировать нахождение ненулевого
    - Для несвязных графов kernel_dim > 1 (равен количеству компонент связности)
    - Может быть неточным для очень маленьких значений (< 1e-10)
    """
    eigvals = eigsh(L, k=kernel_dim+2, which='SA', maxiter=5000)[0]
    return eigvals[kernel_dim] if eigvals[kernel_dim] > 1e-12 else 0.0


def calculate_alpha_inv(self) -> float:
        """
        Вычисляет обратную метрику α_inv = trace / λ_min.
    
        Это "двойственная" метрика к α.
    
        Математически:
        1. Создаём двойственную матрицу: L_α_inv = Ld_inv_sqrt @ Lg @ Ld_inv_sqrt
           где Ld_inv_sqrt = (Ld⁺)^(1/2) - нормирует по трафику
        2. Находим минимальное ненулевое собственное значение λ_min(L_α_inv)
        3. α_inv = trace / λ_min - обратное отношение
    
        Интерпретация:
        - α_inv → ∞ при λ_min → 0 (критическая неустойчивость нагрузки)
    
        Особенность: использует kernel_dim = demands_components_num
        (количество связных компонент в графе трафика), чтобы пропустить
        нулевые собственные значения, соответствующие компонентам.
    
        Returns
        -------
        float
            Обратная метрика устойчивости α_inv (≥ 1, может быть ∞)
        """
        Lg = self.laplacian
        Ld_inv_sqrt = self.demands_pinv_sqrt
        L_alpha_inv = Ld_inv_sqrt @ Lg @ Ld_inv_sqrt
        self.L_alpha_inv = L_alpha_inv
        lam_min = compute_eig_smallest_nonzero(L_alpha_inv, self.demands_components_num)
        tr = float(np.trace(L_alpha_inv))
        self.alpha_inv = tr / lam_min if lam_min != 0.0 else float("inf")
        return tr / lam_min if lam_min != 0.0 else float("inf")
