"""
Mundo de cubos 3D con motor gráfico por GPU (Panda3D)
-----------------------------------------------------
Mismo juego que juego_3d.py (terreno de bloques, árboles, inventario, recetas,
herramientas), pero dibujado con triángulos reales en la tarjeta gráfica:
resolución completa, suavizado de bordes, texturas nítidas y niebla.
La lógica (mundo, jugador, inventario, recetas) se reutiliza de juego_3d.py,
así que este archivo y juego_3d.py deben estar en la misma carpeta.

Requisitos:
    py -m pip install panda3d pygame numpy

Uso:
    py juego_gpu.py
    py juego_gpu.py --ventana 1920x1080
    py juego_gpu.py --pantalla-completa
    py juego_gpu.py --semilla 7 --nivel-agua 0.45 --arboles 1.5
    py juego_gpu.py --benchmark          (mide la generación de la malla, sin ventana)

Controles:
    W A S D ........ mover          Espacio ... saltar (mantén = más alto)
    Shift .......... correr         Mouse ..... girar la cámara (arriba/abajo incluido)
    Clic izquierdo . (mantener) romper bloques / talar árboles
    Clic derecho ... colocar el bloque que llevas en la mano
    1-9 / rueda .... elegir ranura de la barra
    E o R .......... abrir inventario y libro de recetas
    V .............. tercera / primera persona
    F FPS   H ayuda   M soltar/capturar mouse   Esc cerrar / salir
"""

import argparse
import math
import os
import sys
import time
import types

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")     # pygame solo dibuja la interfaz en memoria

import juego_3d as g

try:
    from panda3d.core import (
        loadPrcFileData, WindowProperties, Geom, GeomNode, GeomTriangles, GeomVertexData,
        GeomVertexFormat, GeomVertexArrayFormat, GeomVertexWriter, InternalName, Texture, SamplerState,
        CardMaker, TransparencyAttrib, Fog, LineSegs, AntialiasAttrib, ModifierButtons, ClockObject,
        Point3, Vec3)
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    HAY_PANDA = True
except ImportError:
    HAY_PANDA = False
    ShowBase = object

# --------------------------------------------------------------------------
# Atlas de texturas (todos los bloques en una sola imagen de 4x4 casillas de 16x16)
# --------------------------------------------------------------------------
TS = 4
TILE_PX = 16
TILE = dict(cesped_top=0, cesped_lado=1, tierra=2, arena=3, piedra=4, nieve=5, adoquin=6,
            tronco_top=7, tronco_lado=8, tablones=9, mesa_top=10, mesa_lado=11, hojas=12)

TOP_TILE = np.zeros(10, np.int32)
SIDE_TILE = np.zeros(10, np.int32)


def _asignar(bloque, top, lado):
    TOP_TILE[bloque] = TILE[top]
    SIDE_TILE[bloque] = TILE[lado]


_asignar(g.B_CESPED, "cesped_top", "cesped_lado")
_asignar(g.B_TIERRA, "tierra", "tierra")
_asignar(g.B_ARENA, "arena", "arena")
_asignar(g.B_PIEDRA, "piedra", "piedra")
_asignar(g.B_NIEVE, "nieve", "nieve")
_asignar(g.B_ADOQUIN, "adoquin", "adoquin")
_asignar(g.B_TRONCO, "tronco_top", "tronco_lado")
_asignar(g.B_TABLONES, "tablones", "tablones")
_asignar(g.B_MESA, "mesa_top", "mesa_lado")


def construir_atlas():
    """Imagen RGBA (64, 64, 4) con todas las texturas de bloque (reutiliza las de los iconos)."""
    rng = np.random.default_rng(77)
    t = {}
    c = g._ruido_tex((96, 164, 66), rng, 12)
    c = g._manchas(g._manchas(c, rng, 30, 0.82), rng, 20, 1.14)
    t["cesped_top"] = c
    tierra = g._texturas_bloque(g.I_TIERRA)[0]
    lado = tierra.copy()
    for x in range(16):
        alto = 2 + int(rng.integers(0, 3))
        verde = g._ruido_tex((88, 156, 60), rng, 14)
        lado[:alto, x] = verde[:alto, x]
    t["cesped_lado"] = lado
    t["tierra"] = tierra
    t["arena"] = g._texturas_bloque(g.I_ARENA)[0]
    p = g._ruido_tex((128, 128, 128), rng, 10)
    t["piedra"] = g._manchas(g._manchas(p, rng, 36, 0.78), rng, 12, 1.12)
    t["nieve"] = g._texturas_bloque(g.I_NIEVE)[0]
    t["adoquin"] = g._texturas_bloque(g.I_ADOQUIN)[0]
    t["tronco_top"], t["tronco_lado"] = g._texturas_bloque(g.I_TRONCO)
    t["tablones"] = g._texturas_bloque(g.I_TABLONES)[0]
    t["mesa_top"], t["mesa_lado"] = g._texturas_bloque(g.I_MESA)
    h = g._ruido_tex((58, 136, 50), rng, 20)
    t["hojas"] = g._manchas(g._manchas(h, rng, 60, 0.62), rng, 30, 1.25)

    atlas = np.zeros((TS * TILE_PX, TS * TILE_PX, 4), np.uint8)
    atlas[..., 3] = 255
    for nombre, idx in TILE.items():
        r, col = divmod(idx, TS)
        atlas[r * TILE_PX:(r + 1) * TILE_PX, col * TILE_PX:(col + 1) * TILE_PX, :3] = np.clip(t[nombre], 0, 255)
    return atlas


# --------------------------------------------------------------------------
# Generación de mallas (numpy puro, sin Panda3D)
# Coordenadas de Panda3D: X derecha, Y adelante, Z arriba. El mundo del juego
# usa (x, y, z) con y espejado: la celda (ix, iy) ocupa X in [ix, ix+1] e
# Y in [-(iy+1), -iy].
# --------------------------------------------------------------------------
CH = 16                      # tamaño de cada trozo de mapa (chunk), en bloques
SOMBRA = dict(top=1.0, yp=0.80, yn=0.80, xp=0.62, xn=0.62, bot=0.50)
IDX_TRI = np.array([0, 1, 2, 0, 2, 3])
COLOR_AGUA = (0.20, 0.45, 0.78, 0.60)


def _quad(cara, x0, y0, z0, x1, y1, z1):
    """Esquinas (F, 4, 3) de una cara, en sentido antihorario visto desde fuera."""
    x0, y0, z0, x1, y1, z1 = [a.astype(np.float32) for a in np.broadcast_arrays(*[np.asarray(v) for v in (x0, y0, z0, x1, y1, z1)])]
    if cara == "top":
        v = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    elif cara == "xp":
        v = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
    elif cara == "xn":
        v = [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]
    elif cara == "yp":
        v = [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]
    elif cara == "yn":
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
    else:  # bot
        v = [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]
    return np.stack([np.stack(p, axis=-1) for p in v], axis=-2)


def _uv_tile(tile):
    """Coordenadas de textura (F, 4, 2) de una casilla del atlas (con un pequeño margen)."""
    tile = np.asarray(tile)
    col, fila = tile % TS, tile // TS
    e = 0.6 / (TS * TILE_PX)
    u0, u1 = col / TS + e, (col + 1) / TS - e
    vt, vb = 1 - fila / TS - e, 1 - (fila + 1) / TS + e
    uv = np.empty((len(tile), 4, 2), np.float32)
    uv[:, 0] = np.stack([u0, vb], axis=1)
    uv[:, 1] = np.stack([u1, vb], axis=1)
    uv[:, 2] = np.stack([u1, vt], axis=1)
    uv[:, 3] = np.stack([u0, vt], axis=1)
    return uv


def _triangular(quads, uv, col):
    """Convierte cuadriláteros en triángulos sueltos (6 vértices por cara)."""
    v = quads[:, IDX_TRI].reshape(-1, 3)
    t = uv[:, IDX_TRI].reshape(-1, 2)
    c = np.repeat(col, 6, axis=0)
    return v.astype(np.float32), t.astype(np.float32), c.astype(np.float32)


def _emitir(lista, cara, x0, y0, z0, x1, y1, z1, tile, brillo):
    q = _quad(cara, x0, y0, z0, x1, y1, z1)
    if len(q) == 0:
        return
    col = np.ones((len(q), 4), np.float32)
    col[:, :3] = (SOMBRA[cara] * np.broadcast_to(np.asarray(brillo, np.float32), (len(q),)))[:, None]
    tile = np.broadcast_to(np.asarray(tile), (len(q),))
    lista.append(_triangular(q, _uv_tile(tile), col))


def _unir(lista):
    if not lista:
        z = np.zeros
        return z((0, 3), np.float32), z((0, 2), np.float32), z((0, 4), np.float32)
    return tuple(np.concatenate([p[i] for p in lista]) for i in range(3))


def tipo_capa(mundo, iy, ix, h, d):
    """Tipo del bloque situado `d` capas por debajo de la superficie actual de cada columna."""
    h0 = mundo.h0[iy, ix].astype(np.int32)
    t0 = mundo.tipo0[iy, ix]
    tt = mundo.tipo[iy, ix]
    zb = h - 1 - d
    dn = h0 - 1 - zb
    natural = np.where(dn <= 0, t0, g.B_PIEDRA)
    natural = np.where(((t0 == g.B_CESPED) | (t0 == g.B_TIERRA)) & (dn > 0) & (dn < 4), g.B_TIERRA, natural)
    natural = np.where((t0 == g.B_ARENA) & (dn > 0) & (dn < 3), g.B_ARENA, natural)
    return np.where(zb >= h0, tt, natural).astype(np.int32)


def malla_chunk(mundo, cx, cy):
    """Malla del terreno y del agua de un trozo del mapa. Devuelve (terreno, agua)."""
    n = mundo.n
    ix = np.arange(cx * CH, (cx + 1) * CH)
    iy = np.arange(cy * CH, (cy + 1) * CH)
    ixg, iyg = np.meshgrid(ix, iy)
    H = mundo.alt_int[iyg, ixg].astype(np.int32)
    tipo = mundo.tipo[iyg, ixg].astype(np.int32)
    grano = mundo.grano[iyg * n + ixg]

    piezas = []
    X0 = ixg.ravel().astype(np.float32)
    Y1 = (-iyg).ravel().astype(np.float32)
    Hf = H.ravel().astype(np.float32)
    _emitir(piezas, "top", X0, Y1 - 1, Hf, X0 + 1, Y1, Hf, TOP_TILE[tipo].ravel(), grano.ravel() / 1.07)

    # Paredes laterales: una cara por cada bloque expuesto
    for cara, dx, dy in (("xp", 1, 0), ("xn", -1, 0), ("yp", 0, -1), ("yn", 0, 1)):
        nx, ny = ixg + dx, iyg + dy
        dentro = (nx >= 0) & (nx < n) & (ny >= 0) & (ny < n)
        Hn = np.where(dentro, mundo.alt_int[np.clip(ny, 0, n - 1), np.clip(nx, 0, n - 1)], 0).astype(np.int32)
        dif = H - Hn
        for k in range(1, int(dif.max(initial=0)) + 1):
            m = dif >= k
            if not m.any():
                continue
            iym, ixm, hm = iyg[m], ixg[m], H[m]
            tp = tipo_capa(mundo, iym, ixm, hm, k - 1)
            xa = ixm.astype(np.float32)
            yb = (-iym).astype(np.float32)
            ztop = (hm - (k - 1)).astype(np.float32)
            _emitir(piezas, cara, xa, yb - 1, ztop - 1, xa + 1, yb, ztop, SIDE_TILE[tp], grano[m] / 1.07)
    terreno = _unir(piezas)

    # Agua: una cara en cada columna por debajo del nivel del mar
    m = H.ravel() < mundo.agua_z
    agua = _unir([])
    if m.any():
        q = _quad("top", X0[m], Y1[m] - 1, mundo.agua_z - 0.1, X0[m] + 1, Y1[m], mundo.agua_z - 0.1)
        col = np.tile(np.array(COLOR_AGUA, np.float32), (len(q), 1))
        agua = _triangular(q, np.zeros((len(q), 4, 2), np.float32), col)
    return terreno, agua


def malla_oceano(mundo, margen=700.0):
    """Cuatro rectángulos de agua alrededor del mundo (el horizonte sigue siendo mar)."""
    n, z = float(mundo.n), mundo.agua_z - 0.1
    L = margen
    rects = [(-L, -n - L, 0.0, L), (n, -n - L, n + L, L), (0.0, -n - L, n, -n), (0.0, 0.0, n, L)]
    q = np.concatenate([_quad("top", np.array([a]), np.array([b]), z, np.array([c]), np.array([d]), z) for a, b, c, d in rects])
    col = np.tile(np.array(COLOR_AGUA, np.float32), (len(q), 1))
    return _triangular(q, np.zeros((len(q), 4, 2), np.float32), col)


def malla_arboles(mundo):
    """Todos los árboles activos en una sola malla (tronco + copa de dos niveles)."""
    idx = np.nonzero(mundo.arb_activo)[0]
    if len(idx) == 0:
        return _unir([])
    ix = mundo.arb_x[idx].astype(np.float32)
    iy = mundo.arb_y[idx].astype(np.float32)
    Ht = mundo.arb_h[idx].astype(np.float32)
    base = mundo.alt_int[mundo.arb_y[idx], mundo.arb_x[idx]].astype(np.float32)
    tono = mundo.arb_tono[idx]
    piezas = []
    # tronco
    x0, x1, y0, y1, z0, z1 = ix, ix + 1, -iy - 1, -iy, base, base + Ht
    one = np.ones(len(idx), np.float32)
    for cara in ("xp", "xn", "yp", "yn"):
        _emitir(piezas, cara, x0, y0, z0, x1, y1, z1, TILE["tronco_lado"], one)
    _emitir(piezas, "top", x0, y0, z0, x1, y1, z1, TILE["tronco_top"], one)
    # copas (hojas)
    copas = [(ix - 2, ix + 3, -iy - 3, -iy + 2, base + Ht - 2, base + Ht, 1.0),
             (ix - 1, ix + 2, -iy - 2, -iy + 1, base + Ht, base + Ht + 2, 1.08)]
    for cx0, cx1, cy0, cy1, cz0, cz1, mult in copas:
        for cara in ("xp", "xn", "yp", "yn", "top", "bot"):
            _emitir(piezas, cara, cx0, cy0, cz0, cx1, cy1, cz1, TILE["hojas"], tono * mult / 1.21)
    return _unir(piezas)


def malla_caja(centro, mitad, color):
    """Caja de un solo color (para el personaje). centro y mitad en coordenadas de Panda3D."""
    c = np.array(centro, np.float32)
    m = np.array(mitad, np.float32)
    x0, y0, z0 = c - m
    x1, y1, z1 = c + m
    piezas = []
    for cara in ("top", "xp", "xn", "yp", "yn", "bot"):
        q = _quad(cara, np.array([x0]), np.array([y0]), np.array([z0]), np.array([x1]), np.array([y1]), np.array([z1]))
        col = np.ones((1, 4), np.float32)
        col[0, :3] = np.array(color, np.float32) / 255.0 * SOMBRA[cara]
        piezas.append(_triangular(q, np.zeros((1, 4, 2), np.float32), col))
    return _unir(piezas)


# --------------------------------------------------------------------------
# Enlace con Panda3D
# --------------------------------------------------------------------------
_FORMATO = None


def _formato():
    global _FORMATO
    if _FORMATO is None:
        af = GeomVertexArrayFormat()
        af.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
        af.add_column(InternalName.get_color(), 4, Geom.NT_float32, Geom.C_color)
        af.add_column(InternalName.get_texcoord(), 2, Geom.NT_float32, Geom.C_texcoord)
        f = GeomVertexFormat()
        f.add_array(af)
        _FORMATO = GeomVertexFormat.register_format(f)
    return _FORMATO


def crear_nodo(malla, nombre):
    """Crea un GeomNode de Panda3D a partir de (vértices, uvs, colores). None si está vacío."""
    verts, uvs, cols = malla
    n = len(verts)
    if n == 0:
        return None
    datos = np.empty((n, 9), np.float32)
    datos[:, 0:3] = verts
    datos[:, 3:7] = cols
    datos[:, 7:9] = uvs
    vdata = GeomVertexData(nombre, _formato(), Geom.UH_static)
    vdata.set_num_rows(n)
    escrito = False
    try:                                              # camino rápido 1
        vdata.modify_array(0).modify_handle().set_data(datos.tobytes())
        escrito = vdata.get_num_rows() == n
    except Exception:
        escrito = False
    if not escrito:
        try:                                          # camino rápido 2
            vdata.set_num_rows(n)
            memoryview(vdata.modify_array(0)).cast("B").cast("f")[:] = datos.ravel()
            escrito = True
        except Exception:
            escrito = False
    if not escrito:                                   # camino lento (siempre funciona)
        vdata.set_num_rows(n)
        ev = GeomVertexWriter(vdata, InternalName.get_vertex())
        ec = GeomVertexWriter(vdata, InternalName.get_color())
        et = GeomVertexWriter(vdata, InternalName.get_texcoord())
        for fila in datos.tolist():
            ev.set_data3(fila[0], fila[1], fila[2])
            ec.set_data4(fila[3], fila[4], fila[5], fila[6])
            et.set_data2(fila[7], fila[8])
    prim = GeomTriangles(Geom.UH_static)
    prim.add_consecutive_vertices(0, n)
    prim.close_primitive()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    nodo = GeomNode(nombre)
    nodo.add_geom(geom)
    return nodo


def subir_rgba(tex, datos):
    """Copia bytes RGBA (fila de abajo primero) a una textura de Panda3D."""
    try:
        tex.set_ram_image_as(datos, "RGBA")
    except Exception:
        a = np.frombuffer(datos, np.uint8).reshape(-1, 4)[:, [2, 1, 0, 3]]
        tex.set_ram_image(a.tobytes())


def crear_textura(nombre, ancho, alto, filtro_nitido=True):
    tex = Texture(nombre)
    tex.setup_2d_texture(ancho, alto, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_minfilter(SamplerState.FT_nearest)
    return tex


# --------------------------------------------------------------------------
# Física auxiliar de la cámara
# --------------------------------------------------------------------------
def punto_bloqueado(mundo, x, y, z):
    """¿Hay terreno, tronco o copa de árbol en este punto del mundo? (para que la cámara no los atraviese)."""
    n = mundo.n
    cx, cy = math.floor(x), math.floor(y)
    if cx < 0 or cy < 0 or cx >= n or cy >= n:
        return False
    if z < mundo.altura_celda(cx, cy) + 0.2:
        return True
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            i = mundo.tronco_en(cx + dx, cy + dy)
            if i is None:
                continue
            base = mundo.altura_celda(cx + dx, cy + dy)
            top = base + int(mundo.arb_h[i])
            if dx == 0 and dy == 0 and base <= z < top:
                return True
            if max(abs(dx), abs(dy)) <= 2 and top - 2 <= z < top:
                return True
            if max(abs(dx), abs(dy)) <= 1 and top <= z < top + 2:
                return True
    return False


def camara_tercera(mundo, jug, jz, yaw, pitch, distancia):
    """Cámara orbital: devuelve (x, y, z) en coordenadas del juego, sin atravesar el terreno."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    tx, ty, tz = jug.x, jug.y, jz + 1.4
    libre = 0.6
    t = 0.6
    while t <= distancia:
        px, py, pz = tx - fx * cp * t, ty - fy * cp * t, tz - sp * t
        if punto_bloqueado(mundo, px, py, pz):
            break
        libre = t
        t += 0.2
    libre = max(0.6, libre - 0.15) if libre < distancia else distancia
    return tx - fx * cp * libre, ty - fy * cp * libre, tz - sp * libre, libre


# --------------------------------------------------------------------------
# El juego
# --------------------------------------------------------------------------
TECLAS = ["w", "a", "s", "d", "space", "shift", "lshift", "rshift", "arrow_left", "arrow_right",
          "arrow_up", "arrow_down"]
COLOR_CIELO = (0.55, 0.76, 0.95)
COLOR_BAJO_AGUA = (0.10, 0.30, 0.50)


class Juego(ShowBase):
    def __init__(self, args, mundo):
        ShowBase.__init__(self)
        self.args = args
        self.mundo = mundo
        self.disableMouse()
        self.reloj = ClockObject.getGlobalClock()

        self.ancho = int(self.win.getXSize())
        self.alto = int(self.win.getYSize())
        self.setBackgroundColor(*COLOR_CIELO)
        self.camLens.setFov(75)
        self.camLens.setNearFar(0.12, 420)
        self.render.setAntialias(AntialiasAttrib.M_multisample)

        self.niebla = Fog("niebla")
        self.niebla.setMode(Fog.M_linear)
        self.niebla.setColor(*COLOR_CIELO)
        self.niebla.setLinearRange(args.niebla * 0.45, args.niebla)
        self.render.setFog(self.niebla)
        self.bajo_agua = None
        self._fondo(False)

        # --- lógica del juego (reutilizada de juego_3d.py) ---
        self.jugador = g.Jugador(mundo)
        self.anim = g.Animacion()
        self.inv = g.Inventario()

        # --- interfaz: pygame dibuja en memoria y se muestra como una imagen encima ---
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_mode((1, 1))
        self.fuente = pygame.font.Font(None, 28)
        self.fuente_chica = pygame.font.Font(None, 22)
        self.ui = g.Interfaz(pygame, self.fuente, self.fuente_chica, self.inv)
        self.ui_surf = pygame.Surface((self.ancho, self.alto), pygame.SRCALPHA)
        self.ui_tex = crear_textura("interfaz", self.ancho, self.alto)
        cm = CardMaker("interfaz")
        cm.setFrame(-1, 1, -1, 1)
        self.ui_card = self.render2d.attachNewNode(cm.generate())
        self.ui_card.setTexture(self.ui_tex)
        self.ui_card.setTransparency(TransparencyAttrib.M_alpha)
        self.ui_firma = None
        self.texto_fps = self.fuente.render("FPS: --", True, (255, 255, 255))
        self.ayuda = [self.fuente_chica.render(t, True, (255, 255, 255)) for t in (
            "WASD mover  Espacio saltar  Shift correr  Mouse mirar",
            "Clic izq (mantener): romper/talar   Clic der: colocar   1-9: ranura",
            "E/R: inventario y recetas   V: camara   F: FPS   H: ayuda   Esc: salir")]
        self.mostrar_fps = True
        self.mostrar_ayuda = True

        # --- mundo 3D ---
        self.atlas = crear_textura("atlas", TS * TILE_PX, TS * TILE_PX)
        self.atlas.set_minfilter(SamplerState.FT_nearest_mipmap_linear)
        self.atlas.set_anisotropic_degree(4)
        subir_rgba(self.atlas, np.ascontiguousarray(np.flipud(construir_atlas())).tobytes())
        self.chunks = {}
        self.sucios = set()
        print("Generando el terreno 3D...")
        t0 = time.time()
        for cy in range(mundo.n // CH):
            for cx in range(mundo.n // CH):
                self._construir_chunk(cx, cy)
        self.arboles_np = None
        self._construir_arboles()
        oceano = crear_nodo(malla_oceano(mundo), "oceano")
        self.oceano_np = self.render.attachNewNode(oceano)
        self._estilo_agua(self.oceano_np)
        print(f"Listo en {time.time() - t0:.1f} s")

        self._crear_personaje()
        self._crear_resaltado()

        # --- estado ---
        self.yaw, self.pitch = 0.0, -0.25
        self.tercera = True
        self.teclas = {k: False for k in TECLAS}
        self.salto_pulsado = False
        self.clic_der = False
        self.clic_izq = False
        self.mouse_capturado = not args.sin_mouse
        self.saltar_mouse = True
        self.tiempo = 0.0
        self.mina_clave, self.mina_prog, self.aviso = None, 0.0, 0.0
        self.objetivo = None
        self.dist_camara = 6.0
        self.fov = 75.0
        self._eventos()
        self._ajustar_mouse(self.mouse_capturado)
        self.taskMgr.add(self.actualizar, "actualizar")

    # ---------- configuración ----------
    def _fondo(self, bajo):
        if bajo == self.bajo_agua:
            return
        self.bajo_agua = bajo
        c = COLOR_BAJO_AGUA if bajo else COLOR_CIELO
        self.setBackgroundColor(*c)
        self.niebla.setColor(*c)
        if bajo:
            self.niebla.setLinearRange(0, 26)
        else:
            self.niebla.setLinearRange(self.args.niebla * 0.45, self.args.niebla)

    def _estilo_agua(self, np_):
        np_.setTransparency(TransparencyAttrib.M_alpha)
        np_.setTwoSided(True)
        np_.setDepthWrite(False)

    def _eventos(self):
        mods = ModifierButtons()
        self.buttonThrowers[0].node().setModifierButtons(mods)    # que "shift+w" siga siendo "w"
        for k in TECLAS:
            if k == "space":
                self.accept("space", self._espacio, [True])
                self.accept("space-up", self._espacio, [False])
            else:
                self.accept(k, self._tecla, [k, True])
                self.accept(k + "-up", self._tecla, [k, False])
        self.accept("e", self._inventario)
        self.accept("r", self._inventario)
        self.accept("v", self._camara_v)
        self.accept("f", self._toggle_fps)
        self.accept("h", self._toggle_ayuda)
        self.accept("m", self._toggle_mouse)
        self.accept("escape", self._escape)
        self.accept("mouse1", self._mouse, [1, True])
        self.accept("mouse1-up", self._mouse, [1, False])
        self.accept("mouse3", self._mouse, [3, True])
        self.accept("wheel_up", self._rueda, [1])
        self.accept("wheel_down", self._rueda, [-1])
        for i in range(9):
            self.accept(str(i + 1), self._ranura, [i])

    def _tecla(self, nombre, estado):
        self.teclas[nombre] = estado

    def _espacio(self, estado):
        self.teclas["space"] = estado
        if estado:
            self.salto_pulsado = True

    def _ranura(self, i):
        if not self.ui.abierto:
            self.ui.elegir(i)

    def _rueda(self, d):
        self.ui.rueda(d)

    def _mouse(self, boton, estado):
        if boton == 1:
            self.clic_izq = estado
            if estado and self.ui.abierto:
                self.ui.clic(self.ui_surf, self._pos_mouse(), 1)
        elif boton == 3 and estado:
            if self.ui.abierto:
                self.ui.clic(self.ui_surf, self._pos_mouse(), 3)
            else:
                self.clic_der = True

    def _pos_mouse(self):
        mw = self.mouseWatcherNode
        if not mw.hasMouse():
            return (0, 0)
        m = mw.getMouse()
        return (int((m.getX() + 1) / 2 * self.ancho), int((1 - m.getY()) / 2 * self.alto))

    def _inventario(self):
        if self.ui.abierto:
            self.ui.cerrar()
            self._ajustar_mouse(self.mouse_capturado)
        else:
            self.ui.abrir()
            self._ajustar_mouse(False)

    def _camara_v(self):
        if not self.ui.abierto:
            self.tercera = not self.tercera
            self.pitch = -0.25 if self.tercera else 0.0

    def _toggle_fps(self):
        self.mostrar_fps = not self.mostrar_fps

    def _toggle_ayuda(self):
        self.mostrar_ayuda = not self.mostrar_ayuda

    def _toggle_mouse(self):
        if not self.ui.abierto:
            self.mouse_capturado = not self.mouse_capturado
            self._ajustar_mouse(self.mouse_capturado)

    def _escape(self):
        if self.ui.abierto:
            self._inventario()
        else:
            self.userExit()

    def _ajustar_mouse(self, capturar):
        props = WindowProperties()
        props.setCursorHidden(bool(capturar))
        self.win.requestProperties(props)
        self.saltar_mouse = True

    # ---------- construcción de modelos ----------
    def _construir_chunk(self, cx, cy):
        viejo = self.chunks.pop((cx, cy), None)
        if viejo is not None:
            for nodo in viejo:
                nodo.removeNode()
        terreno, agua = malla_chunk(self.mundo, cx, cy)
        nodos = []
        nt = crear_nodo(terreno, f"chunk{cx}_{cy}")
        if nt is not None:
            np_ = self.render.attachNewNode(nt)
            np_.setTexture(self.atlas)
            nodos.append(np_)
        na = crear_nodo(agua, f"agua{cx}_{cy}")
        if na is not None:
            np_ = self.render.attachNewNode(na)
            self._estilo_agua(np_)
            nodos.append(np_)
        self.chunks[(cx, cy)] = nodos

    def _construir_arboles(self):
        if self.arboles_np is not None:
            self.arboles_np.removeNode()
            self.arboles_np = None
        nodo = crear_nodo(malla_arboles(self.mundo), "arboles")
        if nodo is not None:
            self.arboles_np = self.render.attachNewNode(nodo)
            self.arboles_np.setTexture(self.atlas)

    def _marcar_celda(self, ix, iy):
        n = self.mundo.n
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x, y = ix + dx, iy + dy
                if 0 <= x < n and 0 <= y < n:
                    self.sucios.add((x // CH, y // CH))

    def _crear_personaje(self):
        raiz = self.render.attachNewNode("jugador")
        escala = raiz.attachNewNode("escala")
        cadera = escala.attachNewNode("cadera")
        cadera.setPos(0, 0, g.PIVOTE_CADERA_Z)
        pivotes = {}
        self.mano = None
        for i, (centro, mitad, color, grupo) in enumerate(g.PARTES):
            lx, ly, lz = centro
            hx, hy, hz = mitad
            if grupo in ("pierna_i", "pierna_d"):
                pivote = pivotes.get(grupo)
                if pivote is None:
                    pivote = escala.attachNewNode(grupo)
                    pivote.setPos(ly, 0, g.PIVOTE_CADERA_Z)
                    pivotes[grupo] = pivote
                rel = (0.0, lx, lz - g.PIVOTE_CADERA_Z)
                padre = pivote
            elif grupo in ("brazo_i", "brazo_d"):
                pivote = pivotes.get(grupo)
                if pivote is None:
                    pivote = cadera.attachNewNode(grupo)
                    pivote.setPos(ly, 0, g.PIVOTE_HOMBRO_Z - g.PIVOTE_CADERA_Z)
                    pivotes[grupo] = pivote
                rel = (0.0, lx, lz - g.PIVOTE_HOMBRO_Z)
                padre = pivote
            else:
                rel = (ly, lx, lz - g.PIVOTE_CADERA_Z)
                padre = cadera
            es_mano = i == g.IDX_MANO
            nodo = crear_nodo(malla_caja(rel, (hy, hx, hz), (255, 255, 255) if es_mano else color), f"parte{i}")
            np_ = padre.attachNewNode(nodo)
            if es_mano:
                self.mano = np_
                np_.hide()
        self.modelo = types.SimpleNamespace(raiz=raiz, escala=escala, cadera=cadera, **pivotes)

    def _crear_resaltado(self):
        ls = LineSegs("seleccion")
        ls.setThickness(2.5)
        ls.setColor(0.05, 0.05, 0.05, 1)
        e = 0.01
        a, b = -e, 1 + e
        pts = [(a, a, a), (b, a, a), (b, b, a), (a, b, a), (a, a, b), (b, a, b), (b, b, b), (a, b, b)]
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            ls.moveTo(*pts[i])
            ls.drawTo(*pts[j])
        self.resaltado = self.render.attachNewNode(ls.create())
        self.resaltado.setLightOff(1)
        self.resaltado.hide()

    # ---------- bucle principal ----------
    def actualizar(self, task):
        dt = min(float(self.reloj.getDt()), 0.05)
        self.tiempo += dt
        self.aviso = max(0.0, self.aviso - dt)
        mundo, jug, anim, inv, ui = self.mundo, self.jugador, self.anim, self.inv, self.ui
        jugando = not ui.abierto
        t = self.teclas

        # --- mouse: mirar ---
        if jugando and self.mouse_capturado:
            cx, cy = self.ancho // 2, self.alto // 2
            m = self.win.getPointer(0)
            if self.saltar_mouse:
                self.saltar_mouse = False
            else:
                self.yaw += (m.getX() - cx) * 0.0028
                self.pitch -= (m.getY() - cy) * 0.0022
            self.win.movePointer(0, cx, cy)
        if jugando:
            self.yaw += (t["arrow_right"] - t["arrow_left"]) * 1.9 * dt
            self.pitch += (t["arrow_up"] - t["arrow_down"]) * 1.2 * dt
        self.pitch = max(-1.35, min(1.2, self.pitch))

        # --- jugador ---
        if jugando:
            avance = int(t["w"]) - int(t["s"])
            lateral = int(t["d"]) - int(t["a"])
            correr = t["shift"] or t["lshift"] or t["rshift"]
            mantener = bool(t["space"])
            salto = self.salto_pulsado
        else:
            avance = lateral = 0
            correr = mantener = salto = False
        self.salto_pulsado = False
        jug.actualizar(dt, avance, lateral, self.yaw, correr, salto, mantener)
        lim = mundo.n - 1.0
        if jug.x < 1.0 or jug.x > lim:
            jug.x, jug.vx = min(max(jug.x, 1.0), lim), 0.0
        if jug.y < 1.0 or jug.y > lim:
            jug.y, jug.vy = min(max(jug.y, 1.0), lim), 0.0

        # --- cámara (rotación 3D real) ---
        jz = jug.z + jug.off_visual
        fx, fy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        if self.tercera:
            cx_, cy_, cz_, dist = camara_tercera(mundo, jug, jz, self.yaw, self.pitch, self.dist_camara)
        else:
            cx_, cy_, cz_, dist = jug.x, jug.y, jz + 1.62, 99.0
        self.camera.setPos(cx_, -cy_, cz_)
        self.camera.lookAt(Point3(cx_ + fx * cp, -(cy_ + fy * cp), cz_ + sp))
        fov_obj = 75.0 + 7.0 * min(1.0, jug.velocidad / g.VELOCIDAD_CORRER)
        self.fov += (fov_obj - self.fov) * (1 - math.exp(-dt * 6))
        self.camLens.setFov(self.fov)
        self._fondo(cz_ < mundo.agua_z - 0.05)

        # --- interacción: apuntar, romper y colocar ---
        item = ui.item_en_mano()
        golpeando = False
        self.objetivo = None
        if jugando:
            self.objetivo = g.buscar_objetivo(mundo, jug, self.yaw, math.tan(self.pitch), self.tercera)
            o = self.objetivo
            if o is not None and o["tipo"] == "bloque" and not (0 <= o["cx"] < mundo.n and 0 <= o["cy"] < mundo.n):
                self.objetivo = o = None
            ui.mesa_cerca = mundo.mesa_cerca(jug.x, jug.y, jug.z)
            if self.clic_izq and o is not None:
                clave = (o["tipo"], o.get("idx"), o.get("cx", 0), o.get("cy", 0))
                if clave != self.mina_clave:
                    self.mina_clave, self.mina_prog = clave, 0.0
                t_rot = g.tiempo_rotura(mundo, o, item)
                if t_rot is None:
                    self.mina_prog = 0.0
                    if self.aviso <= 0:
                        ui.mensaje("Necesitas un pico para romper piedra")
                        self.aviso = 1.8
                else:
                    golpeando = True
                    self.mina_prog += dt / t_rot
                    if self.mina_prog >= 1.0:
                        ui.mensaje(g.completar_rotura(mundo, inv, o))
                        if o["tipo"] == "arbol":
                            self._construir_arboles()
                        else:
                            self._marcar_celda(o["cx"], o["cy"])
                        self.mina_prog, self.mina_clave = 0.0, None
            else:
                self.mina_prog, self.mina_clave = 0.0, None
            if self.clic_der and item and "bloque" in g.ITEMS[item]:
                ok, msg = g.intentar_colocar(mundo, jug, o, item)
                if ok:
                    inv.quitar(item, 1)
                    self._marcar_celda(o["cx"], o["cy"])
                elif msg:
                    ui.mensaje(msg)
        else:
            self.mina_prog, self.mina_clave = 0.0, None
        self.clic_der = False
        anim.actualizar(dt, jug, golpeando, self.yaw)
        ui.actualizar(dt)

        # --- reconstruir trozos de mapa modificados ---
        for (cx, cy) in list(self.sucios):
            self._construir_chunk(cx, cy)
        self.sucios.clear()

        self._actualizar_personaje(jug, anim, item, dist)
        self._actualizar_resaltado()
        self._actualizar_interfaz(dt)
        return Task.cont

    def _actualizar_personaje(self, jug, anim, item, dist_camara):
        m = self.modelo
        visible = self.tercera and dist_camara > 1.2
        if not visible:
            m.raiz.hide()
            return
        m.raiz.show()
        a = anim.a
        m.raiz.setPos(jug.x, -jug.y, jug.z + jug.off_visual)
        gi = anim.giro
        m.raiz.setH(math.degrees(math.atan2(-math.cos(gi), -math.sin(gi))))
        esc_z = 1.0 - 0.28 * anim.squash + 0.07 * min(1.0, max(0.0, jug.vz / g.VELOCIDAD_SALTO))
        esc_xy = 1.0 + 0.12 * anim.squash
        rebote = 0.0
        if jug.en_suelo and jug.moviendo and not jug.nadando:
            rebote = 0.07 * abs(math.cos(anim.fase)) * min(1.0, jug.velocidad / 4.0)
        m.escala.setScale(esc_xy, esc_xy, esc_z)
        m.escala.setZ(rebote)
        m.cadera.setP(math.degrees(a["lean"]))
        m.pierna_i.setP(math.degrees(a["pierna_i"]))
        m.pierna_d.setP(math.degrees(a["pierna_d"]))
        m.brazo_i.setP(math.degrees(a["brazo_i"]))
        m.brazo_d.setP(math.degrees(a["brazo_d"]))
        if self.mano is not None:
            if item:
                r, gg, b = g.ITEMS[item]["color"]
                self.mano.setColorScale(r / 255.0, gg / 255.0, b / 255.0, 1)
                self.mano.show()
            else:
                self.mano.hide()

    def _actualizar_resaltado(self):
        o = self.objetivo
        if o is None or self.ui.abierto:
            self.resaltado.hide()
            return
        mundo = self.mundo
        if o["tipo"] == "bloque":
            cx, cy = o["cx"], o["cy"]
            h = mundo.altura_celda(cx, cy)
            self.resaltado.setPos(cx, -(cy + 1), h - 1)
            self.resaltado.setScale(1, 1, 1)
        else:
            i = o["idx"]
            ix, iy = int(mundo.arb_x[i]), int(mundo.arb_y[i])
            self.resaltado.setPos(ix, -(iy + 1), mundo.altura_celda(ix, iy))
            self.resaltado.setScale(1, 1, float(mundo.arb_h[i]))
        self.resaltado.show()

    def _actualizar_interfaz(self, dt):
        ui = self.ui
        fps = float(self.reloj.getAverageFrameRate())
        texto = f"FPS: {fps:5.0f}"
        mouse = self._pos_mouse() if ui.abierto else None
        firma = (tuple(tuple(s) if s else None for s in self.inv.slots), ui.sel, ui.abierto,
                 tuple(m[0] for m in ui.mensajes), ui.t_nombre > 0, round(self.mina_prog * 20),
                 texto if self.mostrar_fps else None, self.mostrar_ayuda, self.tercera,
                 tuple(ui.cursor) if ui.cursor else None, ui.scroll, ui.solo_disp, mouse,
                 ui.mesa_cerca if ui.abierto else None)
        if firma == self.ui_firma:
            return
        self.ui_firma = firma
        pg = self.pg
        surf = self.ui_surf
        surf.fill((0, 0, 0, 0))
        ui.dibujar_hud(surf, not self.tercera, self.mina_prog)
        if ui.abierto:
            ui.dibujar_inventario(surf, mouse)
        if self.mostrar_fps:
            t = self.fuente.render(texto, True, (255, 255, 255))
            fondo = pg.Surface((t.get_width() + 16, t.get_height() + 10), pg.SRCALPHA)
            fondo.fill((0, 0, 0, 150))
            surf.blit(fondo, (8, 8))
            surf.blit(t, (16, 13))
        if self.mostrar_ayuda and not ui.abierto:
            y = 12
            for s in self.ayuda:
                surf.blit(s, (self.ancho - s.get_width() - 12, y))
                y += s.get_height() + 2
        convertir = getattr(pg.image, "tobytes", None) or pg.image.tostring
        subir_rgba(self.ui_tex, convertir(surf, "RGBA", True))


# --------------------------------------------------------------------------
def benchmark(mundo):
    t0 = time.time()
    tri = 0
    caras_max = 0
    for cy in range(mundo.n // CH):
        for cx in range(mundo.n // CH):
            terreno, agua = malla_chunk(mundo, cx, cy)
            tri += (len(terreno[0]) + len(agua[0])) // 3
            caras_max = max(caras_max, len(terreno[0]) // 6)
    dt = time.time() - t0
    t1 = time.time()
    arb = malla_arboles(mundo)
    dt_arb = time.time() - t1
    print(f"Mundo: {mundo.n}x{mundo.n} bloques en {(mundo.n // CH) ** 2} trozos de {CH}x{CH}")
    print(f"Malla del terreno: {dt:.2f} s en total ({dt / (mundo.n // CH) ** 2 * 1000:.1f} ms por trozo)")
    print(f"Triángulos del terreno y agua: {tri:,}   |   árboles: {len(arb[0]) // 3:,} triángulos ({dt_arb * 1000:.0f} ms)")
    print(f"Memoria aproximada de la malla: {(tri * 3 * 36 + len(arb[0]) * 36) / 1e6:.0f} MB")
    t2 = time.time()
    malla_chunk(mundo, 3, 3)
    print(f"Reconstruir un trozo al romper/colocar un bloque: {(time.time() - t2) * 1000:.1f} ms")


def main():
    p = argparse.ArgumentParser(description="Mundo de cubos con motor gráfico por GPU (Panda3D)")
    p.add_argument("--semilla", type=int, default=42, help="Cambia el mundo generado")
    p.add_argument("--tamano", type=int, default=256, help="Tamaño del mundo en bloques (potencia de 2)")
    p.add_argument("--nivel-agua", type=float, default=0.40, help="Nivel del agua (0.15 a 0.65)")
    p.add_argument("--arboles", type=float, default=1.0, help="Cantidad de árboles (0 = ninguno, 2 = bosque denso)")
    p.add_argument("--ventana", default="1280x720", help="Tamaño de la ventana, por ejemplo 1920x1080")
    p.add_argument("--pantalla-completa", action="store_true", help="Abrir a pantalla completa")
    p.add_argument("--niebla", type=float, default=170.0, help="Distancia de dibujado (más alto = ve más lejos)")
    p.add_argument("--suavizado", type=int, default=4, help="Suavizado de bordes (0, 2, 4 u 8; 0 = apagado)")
    p.add_argument("--sin-mouse", action="store_true", help="No capturar el mouse (usa las flechas)")
    p.add_argument("--benchmark", action="store_true", help="Mide la generación de la malla, sin abrir ventana")
    args = p.parse_args()
    args.nivel_agua = min(0.65, max(0.15, args.nivel_agua))

    print("Generando el mundo...")
    mundo = g.Mundo(args.tamano, args.semilla, args.nivel_agua, densidad_arboles=args.arboles)
    if args.benchmark:
        benchmark(mundo)
        return
    if not HAY_PANDA:
        raise SystemExit("Falta Panda3D. Instálalo con:  py -m pip install panda3d pygame")

    try:
        ancho, alto = [int(v) for v in args.ventana.lower().split("x")]
    except ValueError:
        raise SystemExit("Usa --ventana ANCHOxALTO, por ejemplo --ventana 1280x720")
    config = [f"win-size {ancho} {alto}", "window-title Mundo de cubos", "sync-video 1",
              f"fullscreen {1 if args.pantalla_completa else 0}", "show-frame-rate-meter 0"]
    if args.suavizado > 0:
        config += ["framebuffer-multisample 1", f"multisamples {args.suavizado}"]
    loadPrcFileData("", "\n".join(config))

    juego = Juego(args, mundo)
    juego.run()


if __name__ == "__main__":
    main()
