#!/usr/bin/env python
"""MCP server over the typed Opsis tools (src/opsis/tools.py), stdio transport, for an organiser such as the Codex CLI,
Claude Code or any MCP client. Image tools return the PNG as an image content block.
    python scripts/organiser/80_mcp_server.py
Codex CLI config (~/.codex/config.toml):  [mcp_servers.opsis]  command = "<venv>/bin/python"  args = ["scripts/organiser/80_mcp_server.py"]  env = { OPSIS_DATA = "...", PYTHONPATH = "src" }"""
import base64, json
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent, TextContent, ToolAnnotations
RO = dict(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))   # every tool only reads cached data and model weights
from opsis import tools as T
import os
ONLY = set(os.environ["OPSIS_MCP_TOOLS"].split(",")) if os.environ.get("OPSIS_MCP_TOOLS") else None   # an organiser condition can restrict the registered tools
ALIAS = dict(kv.split("=", 1) for kv in os.environ.get("OPSIS_SLIDE_ALIAS", "").split(";") if kv)   # opaque name -> TCGA slide id (the barcode's site code identifies the cancer type)
HIDE = os.environ.get("OPSIS_HIDE_PROJECT") == "1"
def R(slide): return ALIAS.get(slide, slide)
def A(obj, slide):
    """Return the real slide id as the organiser's alias in any result, and drop project and patient when hidden."""
    if HIDE and isinstance(obj, dict): obj = {k: v for k, v in obj.items() if k not in ("project", "patient")}
    alias = next((k for k, v in ALIAS.items() if v == slide), None)
    return json.loads(json.dumps(obj).replace(slide, alias)) if alias else obj
_tool = srv_tool = None
srv = MCPServer("opsis", instructions="Typed pathology readouts over a whole slide image. Call slide_info first, ask_slide for slide level questions, tile_evidence to locate a finding, view_region to look at a location, ask_region for cell and tissue questions about an 80 um field. Probabilities are calibrated; compose composite answers yourself from the primitive answers.")
def tool(**kw):
    def reg(f): return srv.tool(**kw)(f) if (ONLY is None or f.__name__ in ONLY) else f
    return reg
@tool(**RO)
def slide_info(slide: str) -> dict:
    """Project, size, microns per pixel, tile count and the operators available for this slide."""
    return A(T.slide_info(R(slide)), R(slide))
@tool(**RO)
def ask_slide(slide: str, operators: list[str]) -> dict:
    """Slide level typed questions answered by the structured readout (calibrated probabilities over the candidates). Operators: cancer_type, pan_grade, pan_lvi, pan_pni, pan_necrosis, pan_precursor, pan_mitotic, pan_multifocal, pan_infiltrate, tumor_purity_bin, leukocyte_fraction_bin, molecular_subtype."""
    return A(T.ask_slide(R(slide), operators), R(slide))
@tool(**RO)
def tile_evidence(slide: str, finding: str, top_k: int = 6) -> dict:
    """Where a localised finding (necrosis, lvi, pni, precursor) is strongest: slide probability and the top tiles with level 0 coordinates."""
    return A(T.tile_evidence(R(slide), finding, top_k), R(slide))
@tool(**RO)
def ask_region(slide: str, x: int, y: int, operators: list[str], cell_class: str = "tumor") -> dict:
    """Region questions about the 80 um field centred at level 0 (x, y): present, count_bin, cell_frac_bin, more_cells (cell_class tumor | stromal | sTIL | other), tissue_present, tissue_frac_bin, dominant_tissue, stil_score_bin, mitotic_present."""
    return A(T.ask_region(R(slide), x, y, operators, cell_class), R(slide))
@tool(**RO)
def view_region(slide: str, x: int, y: int, size_um: int = 256, mpp: float = 0.5) -> list:
    """An H&E crop centred at level 0 (x, y), size_um microns wide at mpp microns per pixel."""
    r = T.view_region(R(slide), x, y, size_um, mpp); return [TextContent(type="text", text=json.dumps(A({k: v for k, v in r.items() if k != "image_png_base64"}, R(slide)))), ImageContent(type="image", data=r["image_png_base64"], mimeType="image/png")]
@tool(**RO)
def thumbnail(slide: str, max_px: int = 1024) -> list:
    """Low power thumbnail of the slide."""
    r = T.thumbnail(R(slide), max_px); return [TextContent(type="text", text=json.dumps(A({k: v for k, v in r.items() if k != "image_png_base64"}, R(slide)))), ImageContent(type="image", data=r["image_png_base64"], mimeType="image/png")]
if __name__ == "__main__": srv.run("stdio")
