import pygame
import numpy as np
import math

# --- Configurações da Simulação ---
WIDTH, HEIGHT = 1000, 800
BASE_X, BASE_Y = WIDTH // 2, HEIGHT // 2
L1 = 200  # Comprimento do primeiro elo
L2 = 150  # Comprimento do segundo elo
TOLERANCE = 2.0  # Tolerância de parada (em pixels)
DT = 0.05        # Passo de tempo da simulação
KP = 0.5         # Ganho proporcional

# Cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# --- Funções de Cinemática (iguais às anteriores) ---

def forward_kinematics(q1, q2):
    """Calcula a cinemática direta (posição da junta e do efetuador final)."""
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    return (x1, y1), (x2, y2)

def get_jacobian(q1, q2):
    """Calcula a matriz Jacobiana para o efetuador final."""
    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    
    jacobian = np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [ L1 * c1 + L2 * c12,  L2 * c12]
    ])
    return jacobian

# --- Funções Auxiliares do Pygame (iguais) ---

def to_pygame_coords(x, y):
    """Converte coordenadas do robô (base em 0,0) para coordenadas da tela."""
    return int(BASE_X + x), int(BASE_Y - y) # Y é invertido no pygame

def from_pygame_coords(px, py):
    """Converte coordenadas da tela para coordenadas do robô."""
    return float(px - BASE_X), float(BASE_Y - py)

# --- Função Principal da Simulação ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulação 2-DOF (Mínimos Quadrados - Pseudoinversa)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # --- Estado do Robô ---
    q_current = np.array([np.pi / 2, np.pi / 4]) 
    
    # --- Alvo (Target) ---
    target_pos_robot = np.array([100.0, 100.0])
    is_moving = False 

    running = True
    while running:
        # --- Processamento de Eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_px, mouse_py = pygame.mouse.get_pos()
                target_pos_robot = from_pygame_coords(mouse_px, mouse_py)
                is_moving = True 

        # --- Etapa de Controle (Cinemática Inversa com Pseudoinversa) ---
        if is_moving:
            
            j1_pos, ee_pos = forward_kinematics(q_current[0], q_current[1])
            current_pos = np.array(ee_pos)
            
            error = target_pos_robot - current_pos
            distance = np.linalg.norm(error)
            
            if distance < TOLERANCE:
                is_moving = False
            else:
                # --- O CONTROLADOR MAIS SIMPLES ---
                
                # 1. Lei de Controle Proporcional (P)
                # Define a velocidade desejada do efetuador como
                # proporcional ao erro de posição.
                ee_velocity = error * KP
            
                # 2. Calcular o Jacobiano
                J = get_jacobian(q_current[0], q_current[1])
                
                # 3. Solução de Mínimos Quadrados (Pseudoinversa)
                # Encontra a velocidade de junta 'dq' que melhor
                # se aproxima de J * dq = ee_velocity
                try:
                    # J_pinv é a pseudoinversa de J
                    J_pinv = np.linalg.pinv(J)
                    
                    # 4. Aplicar a solução
                    dq = J_pinv @ ee_velocity
                    
                    # 5. Integrar para mover o robô
                    q_current += dq * DT
                    
                except np.linalg.LinAlgError:
                    is_moving = False
                    print("Erro de Álgebra Linear (Singularidade)")

        # --- Etapa de Desenho ---
        screen.fill(BLACK)
        
        help_text = font.render("Clique na tela para definir um novo alvo", 
                                True, WHITE)
        screen.blit(help_text, (10, 10))
        
        max_reach = L1 + L2
        pygame.draw.circle(screen, (50, 0, 0), (BASE_X, BASE_Y), int(max_reach), 2)
        
        target_screen_pos = to_pygame_coords(target_pos_robot[0], target_pos_robot[1])
        pygame.draw.circle(screen, RED, target_screen_pos, 10, 2)
        
        j1_model, ee_model = forward_kinematics(q_current[0], q_current[1])
        
        base_screen = to_pygame_coords(0, 0)
        j1_screen = to_pygame_coords(j1_model[0], j1_model[1])
        ee_screen = to_pygame_coords(ee_model[0], ee_model[1])
        
        pygame.draw.line(screen, WHITE, base_screen, j1_screen, 8)
        pygame.draw.line(screen, WHITE, j1_screen, ee_screen, 8)
        pygame.draw.circle(screen, BLUE, base_screen, 12)
        pygame.draw.circle(screen, BLUE, j1_screen, 10)
        
        ee_color = GREEN if not is_moving else BLUE
        pygame.draw.circle(screen, ee_color, ee_screen, 10)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()