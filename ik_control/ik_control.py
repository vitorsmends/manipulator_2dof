import pygame
import numpy as np
import math

# --- Configurações da Simulação ---
WIDTH, HEIGHT = 1000, 800
BASE_X, BASE_Y = WIDTH // 2, HEIGHT // 2
L1 = 200.0  # Comprimento do primeiro elo
L2 = 150.0  # Comprimento do segundo elo
TOLERANCE = 2.0  # Tolerância (usada apenas para feedback visual)
MOVE_DURATION = 1.5 # Duração da animação em segundos

# Cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (50, 50, 50)

# --- Função de Cinemática Direta (para desenhar) ---

def forward_kinematics(q1, q2):
    """Calcula a cinemática direta (posição da junta e do efetuador final)."""
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    return (x1, y1), (x2, y2)

# --- NOVA FUNÇÃO: Cinemática Inversa Analítica ---

def inverse_kinematics(x, y):
    """
    Calcula a cinemática inversa (nível de posição) usando a Lei dos Cossenos.
    Retorna (q1, q2) para a solução "cotovelo para cima".
    Retorna None se a posição estiver fora de alcance.
    """
    
    # Distância ao quadrado da base ao alvo
    dist_sq = x**2 + y**2
    dist = math.sqrt(dist_sq)
    
    # Alcance máximo e mínimo
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    
    # 1. Verificar se está fora de alcance
    if dist > max_reach or dist < min_reach:
        print(f"Alvo ({x:.1f}, {y:.1f}) fora de alcance.")
        return None
    
    # 2. Calcular q2 (ângulo do cotovelo)
    # Lei dos Cossenos no triângulo (Base, Cotovelo, Efetuador)
    # dist_sq = L1^2 + L2^2 - 2*L1*L2*cos(pi - q2)
    # cos(pi - q2) = -cos(q2)
    # dist_sq = L1^2 + L2^2 + 2*L1*L2*cos(q2)  <-- Cuidado com a definição de q2
    
    # Vamos usar a definição padrão "externa"
    # cos(q2) = (x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2)
    
    # Para garantir robustez numérica, clampamos o valor entre -1 e 1
    cos_q2_arg = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2_arg = max(-1.0, min(1.0, cos_q2_arg)) 
    
    # Solução "cotovelo para cima" (negativo) ou "cotovelo para baixo" (positivo)
    # Vamos escolher "para cima" (negativo) para este exemplo
    q2 = -math.acos(cos_q2_arg)
    
    # 3. Calcular q1 (ângulo da base)
    # q1 = atan2(y, x) - atan2(L2*sin(q2), L1 + L2*cos(q2))
    alpha = math.atan2(y, x)
    beta_arg = (dist_sq + L1**2 - L2**2) / (2 * L1 * dist)
    beta_arg = max(-1.0, min(1.0, beta_arg))
    beta = math.acos(beta_arg)
    
    # q1 para a solução "cotovelo para cima" (-q2)
    q1 = alpha + beta 
    
    # Se tivéssemos escolhido q2 = +acos(...), então q1 = alpha - beta
    
    return np.array([q1, q2])


# --- Funções Auxiliares do Pygame (iguais) ---

def to_pygame_coords(x, y):
    return int(BASE_X + x), int(BASE_Y - y)

def from_pygame_coords(px, py):
    return float(px - BASE_X), float(BASE_Y - py)

# --- Função Principal da Simulação ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulação 2-DOF (Cinemática Inversa Analítica - Nível Posição)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # --- Estado do Robô ---
    q_current = np.array([-np.pi / 2, np.pi / 2]) # Posição inicial
    
    # --- Alvo (Target) ---
    target_pos_robot = np.array([0.0, L1 - L2]) # Posição do alvo inicial
    
    # --- Variáveis de Animação (Interpolação) ---
    q_start = np.copy(q_current)
    q_target = np.copy(q_current)
    is_moving = False 
    trajectory_time = 0.0

    running = True
    while running:
        # --- Processamento de Eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_px, mouse_py = pygame.mouse.get_pos()
                target_pos_robot = from_pygame_coords(mouse_px, mouse_py)
                
                # --- CÁLCULO DE NÍVEL DE POSIÇÃO ---
                # Resolve a Cinemática Inversa UMA VEZ
                q_target_new = inverse_kinematics(target_pos_robot[0], target_pos_robot[1])
                
                if q_target_new is not None:
                    # Inicia a animação
                    q_start = np.copy(q_current)
                    q_target = q_target_new
                    trajectory_time = 0.0
                    is_moving = True
                else:
                    is_moving = False # Alvo fora de alcance

        # --- Etapa de Animação (Interpolação de Posição) ---
        dt = clock.tick(60) / 1000.0 # Delta time em segundos
        
        if is_moving:
            trajectory_time += dt
            t = trajectory_time / MOVE_DURATION # Normaliza o tempo (0.0 a 1.0)
            
            if t >= 1.0:
                t = 1.0
                is_moving = False
                q_current = np.copy(q_target) # Garante que chegou ao alvo
            
            # Interpolação Linear no espaço de juntas
            # Isso é muito mais complexo do que q_current = q_start * (1 - t) + q_target * t
            # porque precisamos encontrar o caminho mais curto entre os ângulos (ex: -170 a 170)
            
            diff = q_target - q_start
            # Envolve a diferença para o caminho mais curto (ex: -pi a +pi)
            diff[0] = (diff[0] + np.pi) % (2 * np.pi) - np.pi
            diff[1] = (diff[1] + np.pi) % (2 * np.pi) - np.pi
            
            q_current = q_start + diff * t
            
        # --- Etapa de Desenho ---
        screen.fill(BLACK)
        
        help_text = font.render("Clique na tela para definir um novo alvo", 
                                True, WHITE)
        screen.blit(help_text, (10, 10))
        
        # Desenhar o círculo de alcance máximo
        max_reach = L1 + L2
        pygame.draw.circle(screen, GRAY, (BASE_X, BASE_Y), int(max_reach), 2)
        
        # Desenhar o alvo
        target_screen_pos = to_pygame_coords(target_pos_robot[0], target_pos_robot[1])
        pygame.draw.circle(screen, RED, target_screen_pos, 10, 2)
        
        # Desenhar o robô
        j1_model, ee_model = forward_kinematics(q_current[0], q_current[1])
        base_screen = to_pygame_coords(0, 0)
        j1_screen = to_pygame_coords(j1_model[0], j1_model[1])
        ee_screen = to_pygame_coords(ee_model[0], ee_model[1])
        
        pygame.draw.line(screen, WHITE, base_screen, j1_screen, 8)
        pygame.draw.line(screen, WHITE, j1_screen, ee_screen, 8)
        pygame.draw.circle(screen, BLUE, base_screen, 12)
        pygame.draw.circle(screen, BLUE, j1_screen, 10)
        
        # Verifica se está perto do alvo (apenas para cor)
        dist_to_target = np.linalg.norm(np.array(ee_model) - target_pos_robot)
        ee_color = GREEN if dist_to_target < TOLERANCE and not is_moving else BLUE
        pygame.draw.circle(screen, ee_color, ee_screen, 10)
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()