from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import pyqtgraph.exporters
import PySide6.QtWidgets as W  # type: ignore
from pandas import DataFrame
from pyqtgraph.functions import mkPen
from PySide6.QtCore import QObject, QSize, Qt, Signal, Slot  # type: ignore
from PySide6.QtGui import QKeySequence  # type: ignore
from scipy.interpolate import RegularGridInterpolator

from chisom._interface.helpers import CyclicGreen, EarthColorMap
from chisom._interface.models import BMUColors, BMUMap, CommonDataModel, FilterModel
from chisom.io.datastores import DatasetBase

# NOTE: Ideally, used QtPixmapCache to store a certain amout of images in memory

pg.setConfigOption("useNumba", True)
pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "w")


class UMap(QObject):
    def __init__(
        self,
        image: npt.NDArray,
        scaling_factor: int = 3,
        layer: int = -1,
        parent: Optional[QObject] = None,
        *args,
        **kwargs,
    ):
        super().__init__(parent=parent)
        self.layer = layer
        self.max_layer = image.shape[0] if image.ndim == 3 else 1
        self.raw_values = image
        self.scaling_factor = scaling_factor  # Default scaling factor
        self.ImageItem = pg.ImageItem()
        self.set_umatrix(self.raw_values)

    def set_umatrix(self, image: npt.NDArray, *args, **kwargs):
        """
        Set the image of the UMap to a new image.
        This is used to update the U-matrix.
        """
        self.selected_values = image[self.layer]
        self.scaled_values = self._interpolate_matrix(
            self.selected_values, self.scaling_factor
        )
        self.ImageItem.setImage(image=self.scaled_values, *args, **kwargs)

    @Slot(int)
    def set_scaling_factor(self, scaling: int):
        """
        Rescale the UMap to a new scaling factor.
        This is used to update the U-matrix with a new scaling factor.
        """
        self._scaling_factor = scaling
        self.scaled_values = self._interpolate_matrix(
            self.selected_values, self.scaling_factor
        )
        self.ImageItem.setImage(image=self.scaled_values)

    @staticmethod
    def _interpolate_matrix(
        matrix: npt.NDArray[np.float32], scaling: int
    ) -> npt.NDArray[np.float32]:
        """
        Interpolates a matrix by a given scaling factor.

        Parameters
        ----------
        matrix : npt.NDArray[np.float32]
            matrix to interpolate
        scaling : int
            scaling factor

        Returns
        -------
        npt.NDArray[np.float32]
            Interpolated umatrix
        """
        # Double the matrix to handle border interpolation
        if scaling < 2:
            return matrix

        fourfold_matrix = np.tile(matrix, (2, 2))

        # create places of known values on grid for interpolation
        rows, cols = fourfold_matrix.shape
        row_steps = np.linspace(0, rows * scaling, rows, dtype=int, endpoint=False)
        col_steps = np.linspace(0, cols * scaling, cols, dtype=int, endpoint=False)

        # create spline function for interpolation
        interpolation_function = RegularGridInterpolator(
            (row_steps, col_steps), fourfold_matrix
        )

        new_rows, new_cols = np.meshgrid(
            range(row_steps.max() + 1),
            range(col_steps.max() + 1),
            indexing="ij",
        )

        # interpolate umatrix
        interpolated_matrix = interpolation_function((new_rows, new_cols))

        # cut interpolated umatrix to orginal projection area
        interpolated_matrix_cut = interpolated_matrix[
            : matrix.shape[0] * scaling, : matrix.shape[1] * scaling
        ]

        # shift interpolated umatrix to have the origins of the original points in the center of an interpolated area
        padding = scaling // 2
        original_rows, original_cols = matrix.shape
        result = np.empty(
            (original_rows * scaling, original_cols * scaling), dtype=np.float32
        )
        # upper left corner
        result[:padding, :padding] = interpolated_matrix_cut[-padding:, -padding:]
        # left edge
        result[padding:, :padding] = interpolated_matrix_cut[:-padding, -padding:]
        # top edge
        result[:padding, padding:] = interpolated_matrix_cut[-padding:, :-padding]
        # rest of map
        result[padding:, padding:] = interpolated_matrix_cut[:-padding, :-padding]

        return result

    @Slot(int)
    def set_layer(self, layer: int):
        """
        Set the layer of the UMap to a new layer.
        This is used to update the U-matrix with a new layer.
        """
        if layer < (self.max_layer * -1) or layer >= self.max_layer:
            raise ValueError(
                f"Layer {layer} out of bounds for UMap with {self.raw_values.shape[0]} layers."
            )
        self.layer = layer
        self.set_umatrix(self.raw_values)


class ImageDelegate(W.QStyledItemDelegate):
    def paint(self, painter, option, index):
        pixmap = index.data(Qt.DecorationRole)
        if pixmap:
            # Center-align the image
            pixmap_rect = option.rect
            pixmap_rect.setWidth(pixmap.width())
            pixmap_rect.setHeight(pixmap.height())
            pixmap_rect.moveCenter(option.rect.center())

            painter.drawPixmap(pixmap_rect, pixmap)
            return
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        pixmap = index.data(Qt.DecorationRole)
        if pixmap:
            return QSize(pixmap.width(), pixmap.height())
        return super().sizeHint(option, index)


class CompoundTable(W.QTableView):
    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.contextMenu = W.QMenu()
        self.contextMenu.addAction("Copy Selection").triggered.connect(self.copySel)
        self.contextMenu.addAction("Copy All").triggered.connect(self.copyAll)
        self.contextMenu.addAction("Save Selection").triggered.connect(self.saveSel)
        self.contextMenu.addAction("Save All").triggered.connect(self.saveAll)

    def setModel(self, model):
        super().setModel(model)

        if hasattr(model, "structure_column_id") and model.structure_column_id != None:
            self.model_has_structure_column = True
            structure_column = model.structure_column_id
            self.structure_info_column_id = model.structure_info_column_id
            self.setItemDelegateForColumn(structure_column, ImageDelegate())

        else:
            self.model_has_structure_column = False

        self.resize_to_contents()

    @Slot()
    def resize_to_contents(self):
        header = self.horizontalHeader()
        self.resizeRowsToContents()
        self.resizeColumnsToContents()

        if self.model_has_structure_column:
            # Now make the structure info column expand to fill remaining space.
            header.setSectionResizeMode(
                self.structure_info_column_id, W.QHeaderView.ResizeMode.Stretch
            )

            # Ensure the last section does not steal extra space unless it's the
            # designated stretch column.
            header.setStretchLastSection(False)
        else:
            header.setSectionResizeMode(-1, W.QHeaderView.ResizeMode.Stretch)
            header.setStretchLastSection(True)

    def serialize(self, useSelection=False):
        """Convert entire table (or just selected area) into tab-separated text values"""
        # Adapted from pyqtgraph TableWidget
        model = self.model()
        if useSelection:
            selection = self.selectedIndexes()
            rows = {index.row() for index in selection}
            columns = {index.column() for index in selection}
        else:
            rows = list(range(model.rowCount()))
            n_columns = model.columnCount()
            if self.model_has_structure_column:
                n_columns -= 1  # Account for the structure column
            columns = list(range(n_columns))
        data = np.empty(
            (len(rows) + 1, len(columns)), dtype="U240"
        )  # account for header row

        for i, c in enumerate(columns):
            data[0, i] = model.headerData(c, Qt.Horizontal, Qt.DisplayRole)

        for i, r in enumerate(rows):
            for j, c in enumerate(columns):
                index = model.index(r, c)
                data[i + 1, j] = model.data(index, Qt.DisplayRole)

        s = ""
        for row in data:
            s += "\t".join(row) + "\n"
        return s

    @Slot()
    def copySel(self):
        """Copy selected data to clipboard."""
        # Adapted from pyqtgraph TableWidget
        W.QApplication.clipboard().setText(self.serialize(useSelection=True))

    @Slot()
    def copyAll(self):
        """Copy all data to clipboard."""
        W.QApplication.clipboard().setText(self.serialize(useSelection=False))

    @Slot()
    def saveSel(self):
        """Save selected data to file."""
        self.save(self.serialize(useSelection=True))

    @Slot()
    def saveAll(self):
        """Save all data to file."""
        self.save(self.serialize(useSelection=False))

    def save(self, data):
        fileName, _ = W.QFileDialog.getSaveFileName(
            self,
            "Save As...",
            "",
            "Tab-separated values (*.tsv)",
        )
        if not fileName:
            return
        with open(fileName, "w") as fd:
            fd.write(data)

    def contextMenuEvent(self, ev):
        self.contextMenu.popup(ev.globalPos())

    def keyPressEvent(self, ev):
        if ev.matches(QKeySequence.StandardKey.Copy):
            ev.accept()
            self.copySel()
        else:
            super().keyPressEvent(ev)


class CatergoryPair(W.QWidget):
    """
    Stores the assosiation between a columns class and the color button instance
    """

    def __init__(self, text):
        super().__init__()
        self.main_layout = W.QHBoxLayout()
        self.category = text
        self.label = W.QLabel(text)
        self.button = pg.ColorButton()
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.button)
        self.setLayout(self.main_layout)


class ColorCategoryWidget(W.QGroupBox):
    cmap_set = Signal(dict)

    def __init__(self, data_columns: List[str], parent=None):
        super().__init__(parent=parent)

        self.currently_selected = None
        self.data_columns = data_columns  # Available columns to color bty
        self.know_columns = {}
        self.category_list = W.QVBoxLayout()
        self.emit_button = W.QPushButton("Set Colors")
        self.setLayout(self.category_list)
        self.emit_button.pressed.connect(self._property_set)

    @Slot(str)
    def select_property(self, name: str):
        """
        Called when a new column to color by is selected and it is categorical
        """
        self.currently_selected = name
        # If this columns has been selected previously, use those values
        if name in self.know_columns:
            self.update_selection(self.know_columns[name])
        # else, create a list of the column available categories and a colorbutton instance
        elif name in self.data_columns:
            color_list = []
            for category in self.data_columns[name][1][1]:
                pair = CatergoryPair(category)
                color_list.append(pair)
            self.know_columns[name] = (
                color_list  # Store for later to keep button instances available
            )
            self.update_selection(self.know_columns[name])
        else:
            raise ValueError("Selected Property unknown")

    def update_selection(self, color_list):
        # Update the currently visible selection in the layout

        # first, remove everything
        while not self.category_list.isEmpty():
            item = self.category_list.itemAt(0)
            if item is not None:
                widget = item.widget()
                self.category_list.removeWidget(widget)
                # only set invisible, as it is the same object as in self.known_columns
                widget.setVisible(False)

        # rebuild layout from list
        for item in color_list:
            item.setVisible(True)
            self.category_list.addWidget(item)
        # add button at the end
        self.category_list.addWidget(self.emit_button)
        self.emit_button.setVisible(True)

    @Slot(bool)
    def _property_set(self):
        cmap = {}
        for widget in self.know_columns[self.currently_selected]:
            cmap[widget.category] = widget.button._color
        self.cmap_set.emit(cmap)


class ControlWidget(W.QGroupBox):
    colormap_changed = Signal(str)
    bmus_toggled = Signal(bool)
    bmus_resized = Signal(int)
    bmus_recolored = Signal(list)

    def __init__(self, cmaps: List[str], data_columns: Dict, parent=None):
        super().__init__("Controls", parent=parent)

        self.bmu_colors: BMUColors = self.parent().bmu_colors
        self.main_layout = W.QVBoxLayout(self)
        self.data_columns = data_columns

        # Colormap
        cmap_layout = W.QHBoxLayout()
        self.cmaps = self.parent().cmaps
        self.cmap_label = W.QLabel("Colormap:")
        self.cmap_selector = W.QComboBox()
        self.cmap_selector.setEditable(False)
        self.cmap_selector.addItems(cmaps)
        self.cmap_selector.currentTextChanged.connect(self.change_colormap)
        cmap_layout.addWidget(self.cmap_label)
        cmap_layout.addWidget(self.cmap_selector)
        self.main_layout.addLayout(cmap_layout)
        self.main_layout.addWidget(W.QFrame(frameShape=W.QFrame.Shape.HLine))

        # BMU control
        bmu_layout = W.QGridLayout()
        self.bmu_visibility_label = W.QLabel("BMUs")
        self.bmu_visibility_toggle = W.QCheckBox("show")
        self.bmu_visibility_toggle.stateChanged.connect(self.toggle_bmus)
        self.bmu_size_label = W.QLabel("Size:")
        self.bmu_size_selector = W.QSpinBox(parent=self)
        self.bmu_size_selector.setRange(1, 200)
        self.bmu_size_selector.setSingleStep(1)
        self.bmu_size_selector.valueChanged.connect(self.resize_bmus)
        self.bmu_color_by_label = W.QLabel("Color by:")
        self.bmu_color_by_selector = W.QComboBox()
        self.bmu_color_by_selector.setEditable(False)
        self.bmu_color_by_selector.addItems(data_columns.keys())
        self.bmu_color_by_selector.textActivated.connect(self.select_property)

        bmu_layout.addWidget(self.bmu_visibility_label, 0, 0)
        bmu_layout.addWidget(self.bmu_visibility_toggle, 0, 1)
        bmu_layout.addWidget(self.bmu_size_label, 1, 0)
        bmu_layout.addWidget(self.bmu_size_selector, 1, 1)
        bmu_layout.addWidget(self.bmu_color_by_label, 2, 0)
        bmu_layout.addWidget(self.bmu_color_by_selector, 2, 1)
        self.main_layout.addLayout(bmu_layout)

        self.category_color = ColorCategoryWidget(data_columns)
        self.category_color.setVisible(False)
        self.main_layout.addWidget(self.category_color)
        self.continous_color = W.QComboBox(self)
        self.continous_color.setEditable(False)
        self.continous_color.addItems(cmaps)
        self.continous_color.setVisible(False)
        self.main_layout.addWidget(self.continous_color)

        self.category_color.cmap_set.connect(self.color_selected_categorical)
        self.continous_color.textActivated.connect(self.color_selected_continous)

        self.main_layout.addStretch()

        self.setLayout(self.main_layout)

    @Slot(str)
    def color_selected_continous(self, cmap: str):
        current_property = self.bmu_color_by_selector.currentText()
        selected_cmap = self.cmaps[cmap]
        self.parent().bmu_colorbar.setVisible(True)
        self.bmu_colors.update_bmu_colors_gradient((current_property, selected_cmap))

    @Slot(dict)
    def color_selected_categorical(self, cmap: dict):
        current_property = self.bmu_color_by_selector.currentText()
        self.parent().bmu_colorbar.setVisible(False)
        self.bmu_colors.update_bmu_colors_categorical((current_property, cmap))

    @Slot(str)
    def select_property(self, name):
        value_type = self.data_columns[name][1][0]
        if value_type == "categorical":
            self.continous_color.setVisible(False)
            self.category_color.select_property(name)
            self.category_color.setVisible(True)
        else:
            self.category_color.setVisible(False)
            self.continous_color.setVisible(True)

    @Slot(int, int)
    def set_bmu_state(self, bmu_state: int, bmu_size: int):
        self.bmu_visibility_toggle.setCheckState(Qt.CheckState(bmu_state))
        self.bmu_size_selector.setValue(bmu_size)

    @Slot(int)
    def toggle_bmus(self, state: int):
        if state == 2:
            self.bmus_toggled.emit(True)
        if state == 0:
            self.bmus_toggled.emit(False)

    @Slot(int)
    def resize_bmus(self, size: int):
        self.bmus_resized.emit(size)

    @Slot(str)
    def change_colormap(self, cmap: str):
        self.cmap_selector.setCurrentText(cmap)
        self.colormap_changed.emit(cmap)


class Roi(pg.PolyLineROI):
    """
    A custom ROI class that emits a signal when the ROI is changed.
    This is used to update the BMU selection based on the ROI.
    """

    roi_changed = Signal()

    def __init__(self, positions=[], closed=True, *args, **kwargs):
        super().__init__(positions=positions, closed=closed, *args, **kwargs)
        self.sigRegionChangeFinished.connect(self._roi_changed)
        self.previous_positions = []

    def addPoint(self, point: npt.NDArray):
        self.previous_positions = [
            tuple(handle["item"].pos()) for handle in self.handles
        ]
        new_positions = self.previous_positions + [tuple(point)]
        self.setPoints(new_positions)

    def clear(self):
        """
        Clear the ROI points.
        """
        self.previous_positions = set()
        super().setPoints(self.previous_positions)

    def _roi_changed(self):
        current_positions = [tuple(handle["item"].pos()) for handle in self.handles]
        if set(current_positions) <= set(self.previous_positions):
            return
        else:
            self.previous_positions = current_positions
            self.roi_changed.emit()


class UpperView(W.QWidget):
    new_bmu_selection = Signal(np.ndarray)

    cmap_list = pg.colormap.listMaps(source="matplotlib")
    cmaps = {}
    for cmap in cmap_list:
        cmaps[cmap] = pg.colormap.get(cmap, source="matplotlib")

    cmaps["Earth"] = EarthColorMap
    cmaps["Cyclic Green"] = CyclicGreen

    def __init__(
        self,
        umap: UMap,
        data_columns: Dict,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.data_model: FilterModel = self.parent().data_model
        self.bmu_map: BMUMap = self.parent().bmu_map
        self.bmu_colors: BMUColors = self.parent().bmu_colors

        self.umap = umap
        self.bmu_pen = mkPen("k", width=1.5)
        self.bmus_points = pg.ScatterPlotItem(
            x=self.bmu_map.bmu_map_coordinates[:, 1],
            y=self.bmu_map.bmu_map_coordinates[:, 0],
            pxMode=True,
            size=10,
            brush=self.bmu_colors.current_colors,
            pen=self.bmu_pen,
        )
        self.bmu_colors.colors_updated.connect(self.set_bmu_colors)

        # Initialize ROI variables
        self.roi = Roi(
            closed=True,
            pen={"color": "k", "width": 4},
            movable=False,
            resizable=False,
        )
        self.roi.roi_changed.connect(self.get_roi)
        self.previous_roi_coords = np.array([], dtype=np.float16)

        self.map_view = pg.ViewBox(invertY=True, lockAspect=True)
        self.map_view.addItem(self.umap.ImageItem)
        self.map_view.addItem(self.bmus_points)
        self.map_view.addItem(self.roi)

        self.map_colorbar = pg.ColorBarItem(
            values=(0, 1),
            limits=(0, 1),
            orientation="vertical",
            interactive=False,
            label="U-height",
        )
        self.map_colorbar.setImageItem(self.umap.ImageItem)

        self.bmu_colorbar = pg.ColorBarItem(
            orientation="vertical",
            interactive=False,
        )
        self.bmu_colorbar.setVisible(False)
        self.bmu_colors.cmap_updated.connect(self.change_bmu_colorbar)

        self.graphic_layout = pg.GraphicsLayoutWidget(parent=self)
        self.graphic_layout.addItem(self.map_view)
        self.graphic_layout.addItem(self.bmu_colorbar)
        self.graphic_layout.addItem(self.map_colorbar)

        self.map_view.scene().sigMouseClicked.connect(self.handle_click)

        self.control = ControlWidget(
            cmaps=list(self.cmaps.keys()), data_columns=data_columns, parent=self
        )
        self.control.colormap_changed.connect(self.change_map_colormap)
        self.control.bmus_resized.connect(self.bmus_points.setSize)
        self.control.bmus_toggled.connect(self.bmus_points.setPointsVisible)
        # self.control.color_widget.properties_set.connect(self.update_bmu_colors)
        # Set the initial colormap to Earth
        self.control.cmap_selector.setCurrentText("Earth")
        self.change_map_colormap("Earth")

        layout = W.QHBoxLayout(self)
        layout.addWidget(self.graphic_layout)
        layout.addWidget(self.control)
        self.setLayout(layout)

    @Slot(list)
    def set_bmus(self, bmu_values: np.ndarray):
        x = bmu_values[:, 1]
        y = bmu_values[:, 0]
        self.bmus_points.setData(x=x, y=y)

    @Slot(str)
    def change_map_colormap(self, cmap: str):
        """Change the colormap of the map and colorbar."""
        if cmap in self.cmaps:
            self.map_colorbar.setColorMap(self.cmaps[cmap])
        else:
            print(f"Colormap {cmap} not found.")

    @Slot(list)
    def change_bmu_colorbar(self, info: list):
        property_cmap, minimum, maximum, property_name = info
        self.bmu_colorbar.setColorMap(property_cmap)
        self.bmu_colorbar.setLevels(low=minimum, high=maximum)
        self.bmu_colorbar.setLabel(axis="left", text=property_name)

    @Slot(list)
    def set_bmu_colors(self, colors):
        self.bmus_points.setBrush(colors)
        self.bmus_points.setPen(self.bmu_pen)

    def handle_click(self, event):
        # Get click position in the view coordinates
        pos = self.map_view.mapSceneToView(event.scenePos())
        modifier = event.modifiers()

        if modifier == Qt.KeyboardModifier.ControlModifier:
            self.roi.addPoint((pos.x(), pos.y()))
        else:
            self.roi.clear()

    @Slot()
    def get_roi(self):
        data, roi_coords = self.roi.getArrayRegion(
            self.umap.ImageItem.image,
            self.umap.ImageItem,
            returnMappedCoords=True,
        )

        roi_coords = np.transpose(roi_coords, (1, 2, 0))
        roi_coords = np.floor(roi_coords)  # round to nearest full pixel
        mask = np.where(data > 0)
        roi_coords = roi_coords[mask[0], mask[1]]
        if len(roi_coords) > 0:
            self.new_bmu_selection.emit(roi_coords)


class SelectionView(W.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.installEventFilter(self)
        # self.setOrientation(Qt.Orientation.Horizontal)

        self.data_model: FilterModel = self.parent().data_model

        self.table = CompoundTable(parent=self)
        self.table.setModel(self.data_model)
        self.data_model.selection_changed.connect(self.table.resize_to_contents)
        layout = W.QHBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)


class MainView(W.QSplitter):
    def __init__(
        self,
        umap: UMap,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.installEventFilter(self)
        self.setOrientation(Qt.Orientation.Vertical)

        self.data_model: FilterModel = self.parent().data_model
        self.bmu_map: BMUMap = self.parent().bmu_map
        self.bmu_colors: BMUColors = self.parent().bmu_colors

        self.upper_view = UpperView(
            umap=umap,
            data_columns=self.data_model.columns_with_properties,
            parent=self,
        )
        self.data_view = SelectionView(parent=self)

        self.addWidget(self.upper_view)

        self.addWidget(self.data_view)

        self.upper_view.new_bmu_selection.connect(self.new_bmu_selection)

    @Slot(np.ndarray)
    def new_bmu_selection(self, selection_coords: npt.NDArray):
        # Get the BMU indices and data indices from the map coordinates
        scatter_indices, data_indices = self.bmu_map.get_bmu_info_from_map_coordinates(
            selection_coords
        )
        self.data_model.set_selected_rows(data_indices)
        # self.upper_view.bmus_points.setSelected(scatter_indices)


class MainSomWindow(W.QMainWindow):
    def __init__(
        self,
        umatrix: npt.NDArray,
        bmu_coordinates: Optional[npt.NDArray],
        data: Union[DatasetBase, DataFrame],
        structure_info_column: Optional[str],
        scaling_factor: int,
    ):
        super().__init__()
        self.setMinimumSize(QSize(800, 600))
        self.setWindowTitle("ChI-SOM")

        self.base_model: Optional[CommonDataModel] = (
            CommonDataModel(
                data, structure_info_column=structure_info_column, parent=self
            )
            if data is not None
            else None
        )
        self.data_model = FilterModel(self.base_model, parent=self)

        if umatrix.ndim == 2:
            umatrix = umatrix[np.newaxis, :, :]
        elif umatrix.ndim != 3:
            raise ValueError("U-matrix must be 2D or 3D")

        self.umap = UMap(umatrix, scaling_factor=scaling_factor, parent=self)
        self.bmu_map = BMUMap(
            bmu_raw_coordinates=bmu_coordinates,
            data_index=data.index,
            scaling_factor=scaling_factor,
        )
        self.bmu_colors = BMUColors(self.data_model, self.bmu_map)

        self.main_view = MainView(
            umap=self.umap,
            parent=self,
        )

        self.setCentralWidget(self.main_view)

        # Set BMUs and trigger for later chages in scaling
        self.bmu_map.map_bmu_coordinates_changed.connect(
            self.main_view.upper_view.set_bmus
        )
        self.main_view.upper_view.control.set_bmu_state(self.bmu_map.bmu_state, 10)


def start_chisom_viewer(
    umatrix: npt.NDArray,
    bmu_coordinates: npt.NDArray,
    data: Union[DatasetBase, DataFrame],
    structure_info_column: Optional[str] = None,
    scaling_factor: int = 3,
):
    """Start the GUI interface

    Parameters
    ----------
    umatrix
        U-Matrix of the SOM.
    bmu_coordinates
        Coordinates of the BMU to the data points.
    data
        Additional data to the data points. Will be renderd in the tabel view and used for coloring of BMUs.
    structure_info_column
        With chemical dataset the column with this index supplies the SMILES to render the molecule, by default None.
    scaling_factor
        Will scale the U-Matrix by this factor ands interpolation for an anti-aliased view, by default 3.
    """

    app = pg.mkQApp("ChI-SOM Viewer")

    window = MainSomWindow(
        umatrix, bmu_coordinates, data, structure_info_column, scaling_factor
    )
    window.show()

    app.exec()
