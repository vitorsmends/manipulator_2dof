import pygame
import numpy as np
import math

# --- Configurações da Simulação ---
WIDTH, HEIGHT = 1000, 800
BASE_X, BASE_Y = WIDTH // 2, HEIGHT // 2
L1 = 200  # Comprimento do primeiro elo
L2 = 150  # Comprimento do segundo elo
TOLERANCE = 2.0  # Tolerância de parada (em pixels)
DT = 0.05        # Passo de tempo da simulação (para integração)

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

# --- Funções Auxiliares do Pygame ---

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
    pygame.display.set_caption("Simulação 2-DOF (Damped Least Squares)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # --- Estado do Robô ---
    # q_current é o estado atual das juntas [q1, q2]
    q_current = np.array([np.pi / 2, np.pi / 4]) 
    
    # --- Alvo (Target) ---
    # Posição alvo inicial (em coordenadas do robô)
    target_pos_robot = np.array([100.0, 100.0])
    is_moving = False # Flag para controlar se o robô deve se mover

    running = True
    while running:
        # --- Processamento de Eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Define um novo alvo ao clicar
                mouse_px, mouse_py = pygame.mouse.get_pos()
                target_pos_robot = from_pygame_coords(mouse_px, mouse_py)
                is_moving = True # Inicia o movimento

        # --- Etapa de Controle (Cinemática Inversa Robusta) ---
        if is_moving:
            
            # 1. Calcular a posição atual do efetuador final (FK)
            j1_pos, ee_pos = forward_kinematics(q_current[0], q_current[1])
            current_pos = np.array(ee_pos)
            
            # 2. Calcular o erro (vetor do efetuador até o alvo)
            error = target_pos_robot - current_pos
            distance = np.linalg.norm(error)
            
            # 3. Verificar condição de parada
            if distance < TOLERANCE:
                is_moving = False
            else:
                # --- Implementação do Damped Least Squares (DLS) ---
                
                # Ganho proporcional para definir a velocidade desejada
                Kp = 0.5 
                ee_velocity = error * Kp # Velocidade desejada do efetuador
            
                # 4. Calcular o Jacobiano na configuração atual
                J = get_jacobian(q_current[0], q_current[1])
                J_T = J.T # Jacobiano Transposto
                
                # 5. Fator de amortecimento (lambda)
                # Pode ser constante ou dinâmico
                lambda_sq = 0.1**2 # lambda ao quadrado
                
                # 6. Resolver a equação DLS: dq = J_T * inv(J*J_T + lambda^2*I) * x_vel
                try:
                    identity_matrix = np.identity(2)
                    
                    # (J*J_T + lambda^2*I)
                    term_to_invert = J @ J_T + lambda_sq * identity_matrix
                    
                    # inv(J*J_T + lambda^2*I)
                    inv_term = np.linalg.inv(term_to_invert)
                    
                    # J_dls = J_T * inv(J*J_T + lambda^2*I)
                    J_dls = J_T @ inv_term
                    
                    # 7. Calcular a velocidade das juntas necessárias
                    dq = J_dls @ ee_velocity # dq é [dq1, dq2]
                    
                    # 8. Atualizar os ângulos das juntas (Integração de Euler)
                    q_current += dq * DT
                    
                except np.linalg.LinAlgError:
                    # Caso a matriz seja singular mesmo com DLS (muito raro)
                    is_moving = False
                    print("Erro de Álgebra Linear (Singularidade)")

        # --- Etapa de Desenho ---
        screen.fill(BLACK)
        
        # Desenhar texto de ajuda
        help_text = font.render("Clique na tela para definir um novo alvo", 
                                True, WHITE)
        screen.blit(help_text, (10, 10))
        
        # Desenhar o alvo (círculo vermelho)
        target_screen_pos = to_pygame_coords(target_pos_robot[0], target_pos_robot[1])
        pygame.draw.circle(screen, RED, target_screen_pos, 10, 2)
        
        # Calcular posições atuais do robô para desenhar
        j1_model, ee_model = forward_kinematics(q_current[0], q_current[1])
        
        base_screen = to_pygame_coords(0, 0)
        j1_screen = to_pygame_coords(j1_model[0], j1_model[1])
        ee_screen = to_pygame_coords(ee_model[0], ee_model[1])
        
        # Desenhar os elos
        pygame.draw.line(screen, WHITE, base_screen, j1_screen, 8) # Elo 1
        pygame.draw.line(screen, WHITE, j1_screen, ee_screen, 8)   # Elo 2
        
        # Desenhar as juntas
        pygame.draw.circle(screen, BLUE, base_screen, 12) # Base
        pygame.draw.circle(screen, BLUE, j1_screen, 10)  # Cotovelo
        
        # Mudar a cor do efetuador se ele alcançou o alvo
        ee_color = GREEN if not is_moving else BLUE
        pygame.draw.circle(screen, ee_color, ee_screen, 10) # Efetuador final
        
        # Atualizar a tela
        pygame.display.flip()
        
        # Controlar o FPS
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()