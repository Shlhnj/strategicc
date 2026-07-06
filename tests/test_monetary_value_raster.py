"""
tests/test_monetary_value_raster.py  —  v3.12
Unit tests for accounting.outputs.save_monetary_value_raster (Mode C
genuine per-pixel valuation raster).
"""

import pytest
import numpy as np
from PIL import Image

from strategicc.accounting.csv_loader import EcosystemService
from strategicc.accounting.outputs import save_monetary_value_raster
from strategicc.io.csv_loader import StateClass


@pytest.fixture
def classes():
    return {
        1: StateClass(id=1, name="Mangrove", full_name="Mangrove:All", color=(255, 0, 128, 0)),
        2: StateClass(id=2, name="Water_body", full_name="Water_body:All", color=(255, 0, 0, 255)),
    }


@pytest.fixture
def mode_c_service():
    return EcosystemService(
        state_class="Mangrove", service_name="Carbon Storage", service_type="Regulating",
        value_per_unit_area=238333.0, currency="IDR", physical_unit="MgC",
        physical_per_unit_area=None, stockflow_source="stock:Biomass",
    )


def _write_raster(path, arr, mode):
    # Minimal PixelScaleTag (33550) so read_tiff()/_pixel_area_ha() doesn't
    # KeyError -- value itself is irrelevant to these tests (only stock/
    # LULC arrays and the class mask matter here, not pixel area).
    tiffinfo = {33550: (0.0001, 0.0001, 0.0)}
    Image.fromarray(arr, mode=mode).save(str(path), tiffinfo=tiffinfo)


def test_save_monetary_value_raster_prices_only_matching_class(
    tmp_path, classes, mode_c_service
):
    lulc = np.array([[1, 1, 2], [2, 1, 2]], dtype=np.uint8)
    stock = np.array([[10.0, 20.0, 5.0], [5.0, 30.0, 5.0]], dtype=np.float32)

    lulc_path = tmp_path / "lulc_2020.tif"
    stock_path = tmp_path / "stock_2020.tif"
    _write_raster(lulc_path, lulc, "L")
    _write_raster(stock_path, stock, "F")

    out_path = tmp_path / "value_2020.tif"
    save_monetary_value_raster(
        stock_raster_path=stock_path, lulc_raster_path=lulc_path,
        service=mode_c_service, classes=classes, out_path=out_path,
        nodata_value=-1.0,
    )

    result = np.array(Image.open(str(out_path)))

    # Mangrove (class 1) pixels priced: stock * price
    assert result[0, 0] == pytest.approx(10.0 * 238333.0)
    assert result[0, 1] == pytest.approx(20.0 * 238333.0)
    assert result[1, 1] == pytest.approx(30.0 * 238333.0)

    # Water_body (class 2) pixels get nodata -- Mangrove-only service
    assert result[0, 2] == pytest.approx(-1.0)
    assert result[1, 0] == pytest.approx(-1.0)
    assert result[1, 2] == pytest.approx(-1.0)


def test_save_monetary_value_raster_rejects_non_mode_c(tmp_path, classes):
    svc = EcosystemService(
        state_class="Mangrove", service_name="Tourism", service_type="Cultural",
        value_per_unit_area=5000, currency="IDR", physical_unit=None,
        physical_per_unit_area=None, stockflow_source=None,
    )
    lulc_path = tmp_path / "lulc.tif"
    stock_path = tmp_path / "stock.tif"
    _write_raster(lulc_path, np.ones((2, 2), dtype=np.uint8), "L")
    _write_raster(stock_path, np.ones((2, 2), dtype=np.float32), "F")

    with pytest.raises(ValueError, match="Mode C"):
        save_monetary_value_raster(
            stock_raster_path=stock_path, lulc_raster_path=lulc_path,
            service=svc, classes=classes, out_path=tmp_path / "out.tif",
        )


def test_save_monetary_value_raster_shape_mismatch_raises(
    tmp_path, classes, mode_c_service
):
    lulc_path = tmp_path / "lulc.tif"
    stock_path = tmp_path / "stock.tif"
    _write_raster(lulc_path, np.ones((2, 2), dtype=np.uint8), "L")
    _write_raster(stock_path, np.ones((3, 3), dtype=np.float32), "F")

    with pytest.raises(ValueError, match="shape"):
        save_monetary_value_raster(
            stock_raster_path=stock_path, lulc_raster_path=lulc_path,
            service=mode_c_service, classes=classes, out_path=tmp_path / "out.tif",
        )


def test_save_monetary_value_raster_unknown_state_class_raises(
    tmp_path, classes
):
    svc = EcosystemService(
        state_class="Nonexistent_class", service_name="X", service_type="Regulating",
        value_per_unit_area=1.0, currency="IDR", physical_unit="MgC",
        physical_per_unit_area=None, stockflow_source="stock:Biomass",
    )
    lulc_path = tmp_path / "lulc.tif"
    stock_path = tmp_path / "stock.tif"
    _write_raster(lulc_path, np.ones((2, 2), dtype=np.uint8), "L")
    _write_raster(stock_path, np.ones((2, 2), dtype=np.float32), "F")

    with pytest.raises(ValueError, match="not found in classes"):
        save_monetary_value_raster(
            stock_raster_path=stock_path, lulc_raster_path=lulc_path,
            service=svc, classes=classes, out_path=tmp_path / "out.tif",
        )
