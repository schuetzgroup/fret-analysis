// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs as QQDialogs
import QtQuick.Layouts
import SdtGui as Sdt
import FRETInspector


ApplicationWindow {
    id: window
    property alias backend: backend

    visible: true
    title: "FRET inspector"

    RowLayout {
        id: rootLayout

        anchors.fill: parent
        anchors.margins: 5

        ColumnLayout {
            Layout.alignment: Qt.AlignTop

            Row {
                ToolButton {
                    icon.name: "document-open"
                    onClicked: {
                        saveFileDialog.fileMode = QQDialogs.FileDialog.OpenFile
                        saveFileDialog.open()
                    }
                }
                ToolButton {
                    icon.name: "document-save"
                    onClicked: {
                        // saveFileDialog.fileMode = QQDialogs.FileDialog.SaveFile
                        // saveFileDialog.open()
                        backend.save(false)
                    }
                }
                ToolButton {
                    icon.name: "x-office-spreadsheet"
                    onClicked: { backend.save(true) }
                }
            }
            GroupBox {
                title: "display options"

                Layout.fillWidth: true
                Layout.minimumWidth: displayOptLayout.implicitWidth

                GridLayout {
                    id: displayOptLayout

                    columns: 3
                    anchors.fill: parent

                    Switch {
                        id: showManuallyFilteredCheck
                        text: "include rejected"
                        checked: false
                        Layout.columnSpan: 3
                    }
                    Binding {
                        target: fileSel.particles != null ? fileSel.particles : null
                        property: "showManuallyFiltered"
                        value: showManuallyFilteredCheck.checked
                    }
                    Label {
                        text: "track length"
                        Layout.rowSpan: 2
                        Layout.fillWidth: true
                    }
                    Label { text: "min." }
                    Sdt.EditableSpinBox {
                        id: minTrackLenSel
                        to: Sdt.Sdt.intMax
                    }
                    Label { text: "max." }
                    Sdt.EditableSpinBox {
                        id: maxTrackLenSel
                        to: Sdt.Sdt.intMax
                        value: 10000
                    }
                    Binding {
                        target: fileSel.particles != null ? fileSel.particles : null
                        property: "trackLengthRange"
                        value: [minTrackLenSel.value, maxTrackLenSel.value]
                    }
                    Switch {
                        id: hideInterpolatedCheck
                        text: "hide interpolated datapoints"
                        checked: false
                        Layout.columnSpan: 3
                    }
                    Binding {
                        target: fileSel.particles != null ? fileSel.particles : null
                        property: "hideInterpolated"
                        value: hideInterpolatedCheck.checked
                    }
                    Switch {
                        id: showLocCheck
                        text: "show localizations"
                        checked: true
                        Layout.columnSpan: 3
                    }
                    Switch {
                        id: scatterCheck
                        text: "E-vs.-S scatter plot"
                        checked: true
                        Layout.columnSpan: 3
                        onCheckedChanged: {
                            backend.plot(particleSel.smData, checked)
                        }
                    }
                }
            }
            GroupBox {
                title: "frame navigation"

                Layout.fillWidth: true

                GridLayout {
                    columns: 2
                    anchors.fill: parent
                    Switch {
                        id: autoFirstCheck
                        text: "auto. go to first frame"
                        checked: true
                        Layout.columnSpan: 2
                        Layout.fillWidth: true
                    }
                    ToolButton {
                        icon.name: "go-first"
                        Layout.fillWidth: true
                        onClicked: {
                            frameSel.value = backend.firstFrame(particleSel.dTrackData)
                        }
                    }
                    ToolButton {
                        icon.name: "go-last"
                        Layout.fillWidth: true
                        onClicked: {
                            frameSel.value = backend.lastFrame(particleSel.dTrackData)
                        }
                    }
                }
            }
            GroupBox {
                id: particleSelGroup

                title: "particle selection"

                function previousParticle() {
                    if (particleSel.currentIndex > 0) {
                        particleSel.decrementCurrentIndex()
                        return
                    }
                    if (fileSel.currentIndex > 0) {
                        fileSel.decrementCurrentIndex()
                        particleSel.currentIndex = particleSel.count - 1
                        return
                    }
                    datasetSel.decrementCurrentIndex()
                    fileSel.currentIndex = fileSel.count - 1
                    particleSel.currentIndex = particleSel.count - 1
                }

                function nextParticle() {
                    if (particleSel.currentIndex < particleSel.count - 1) {
                        particleSel.incrementCurrentIndex()
                        return
                    }
                    if (fileSel.currentIndex < fileSel.count - 1) {
                        fileSel.incrementCurrentIndex()
                        return
                    }
                    datasetSel.incrementCurrentIndex()
                }

                Layout.fillWidth: true

                GridLayout {
                    columns: 2
                    anchors.fill: parent

                    ToolButton {
                        icon.name: "go-previous"
                        Layout.fillWidth: true
                        enabled: (datasetSel.currentIndex > 0 || fileSel.currentIndex > 0 ||
                                  particleSel.currentIndex > 0)
                        onClicked: { particleSelGroup.previousParticle() }
                    }
                    ToolButton {
                        icon.name: "go-next"
                        Layout.fillWidth: true
                        enabled: (datasetSel.currentIndex < datasetSel.count - 1 ||
                                  fileSel.currentIndex < fileSel.count - 1 ||
                                  particleSel.currentIndex < particleSel.count - 1)
                        onClicked: { particleSelGroup.nextParticle() }
                    }
                    ToolButton {
                        icon.name: "dialog-ok-apply"
                        icon.color: "green"
                        Layout.fillWidth: true
                        onClicked: {
                            // check whether we need to manually go to next
                            // particle
                            fileSel.particles.manuallyFilterTrack(particleSel.currentIndex, false)
                            particleSelGroup.nextParticle()
                        }
                    }
                    ToolButton {
                        icon.name: "dialog-cancel"
                        icon.color: "red"
                        Layout.fillWidth: true
                        onClicked: {
                            var goNext = false
                            if (showManuallyFilteredCheck.checked ||
                                    particleSel.currentIndex == particleSel.count - 1)
                                goNext = true
                            fileSel.particles.manuallyFilterTrack(particleSel.currentIndex, true)
                            if (goNext)
                                particleSelGroup.nextParticle()
                        }
                    }
                    Label {
                        Layout.columnSpan: 2
                        Layout.alignment: Qt.AlignCenter
                        text: "filter: " + filterText(particleSel.manualFilter)

                        function filterText(n) {
                            switch (n) {
                                case 0:
                                    return "particle was accepted"
                                case 1:
                                    return "particle was rejected"
                            }
                            return "no decision yet"
                        }
                    }
                }
            }
            Item { Layout.fillHeight: true}
        }
        ColumnLayout {
            RowLayout {
                Label { text: "dataset" }
                Sdt.DatasetSelector {
                    id: datasetSel

                    Layout.fillWidth: true
                    datasets: backend.datasets
                }
                Item { width: 5 }
                Label { text: "file" }
                ComboBox {
                    id: fileSel

                    property var ddImg: null
                    property var daImg: null
                    property var aaImg: null
                    property var particles: null

                    model: datasetSel.currentDataset
                    textRole: "display"
                    valueRole: "id"
                    Layout.fillWidth: true

                    function _setProps() {
                        if (model) {
                            ddImg = model.get(currentIndex, "ddImg")
                            daImg = model.get(currentIndex, "daImg")
                            aaImg = model.get(currentIndex, "aaImg")
                            particles = model.get(currentIndex, "particles")
                        } else {
                            ddImg = null
                            daImg = null
                            aaImg = null
                            particles = null
                        }
                    }

                    onModelChanged: _setProps()
                    onCurrentIndexChanged: _setProps()
                }
                Item { width: 5 }
                Label { text: "particle" }
                ComboBox {
                    id: particleSel

                    property int number: -1
                    property var smData: null
                    property var dTrackData: null
                    property var aTrackData: null
                    property var manualFilter: null

                    property var _plotDummy: backend.plot(smData, scatterCheck.checked)

                    textRole: "display"
                    model: fileSel.particles

                    function _setProps() {
                        if (model) {
                            number = model.get(currentIndex, "number")
                            smData = model.get(currentIndex, "smData")
                            dTrackData = model.get(currentIndex, "dTrackData")
                            aTrackData = model.get(currentIndex, "aTrackData")
                            manualFilter = model.get(currentIndex, "manualFilter")
                        } else {
                            number = -1
                            smData = null
                            dTrackData = null
                            aTrackData = null
                            manualFilter = null
                        }
                    }

                    onModelChanged: _setProps()
                    onCurrentIndexChanged: _setProps()
                    Connections {
                        target: particleSel.model != null ? particleSel.model : null
                        function onItemsChanged() { particleSel._setProps() }
                        function onCountChanged() { particleSel._setProps() }
                    }
                    onDTrackDataChanged: {
                        if (autoFirstCheck.checked)
                            frameSel.value = backend.firstFrame(dTrackData)
                    }
                }
                Item { width: 5 }
                Label { text: "frame" }
                Sdt.EditableSpinBox {
                    id: frameSel
                    to: backend.frameCount(fileSel.ddImg)
                }
            }
            GridLayout {
                columns: 3

                Label {
                    text: "donor → donor"
                    Layout.alignment: Qt.AlignCenter
                }
                Label {
                    text: "donor → acceptor"
                    Layout.alignment: Qt.AlignCenter
                }
                Label {
                    text: "acceptor → acceptor"
                    Layout.alignment: Qt.AlignCenter
                }
                Sdt.ImageDisplay {
                    image: backend.image(fileSel.ddImg, frameSel.value)
                    overlays: Sdt.TrackDisplay {
                        trackData: particleSel.dTrackData
                        currentFrame: frameSel.value
                        visible: showLocCheck.checked
                    }
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                Sdt.ImageDisplay {
                    image: backend.image(fileSel.daImg, frameSel.value)
                    overlays: Sdt.TrackDisplay {
                        trackData: particleSel.aTrackData
                        currentFrame: frameSel.value
                        visible: showLocCheck.checked
                    }
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                Sdt.ImageDisplay {
                    image: backend.image(fileSel.aaImg, frameSel.value)
                    overlays: Sdt.TrackDisplay {
                        trackData: particleSel.aTrackData
                        currentFrame: frameSel.value
                        visible: showLocCheck.checked
                    }
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
            }
            Item { height: 5 }
            RowLayout {
                Sdt.FigureCanvasAgg {
                    id: figCanvas
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 250
                }
            }
        }
    }
    Backend {
        id: backend
        figureCanvas: figCanvas
        minTrackLength: minTrackLenSel.value
        onMinTrackLengthChanged: { minTrackLenSel.value = minTrackLength }
        maxTrackLength: maxTrackLenSel.value
        onMaxTrackLengthChanged: { maxTrackLenSel.value = maxTrackLength }
    }
    Settings {
        id: settings
        category: "Window"
        property int width: { width = rootLayout.implicitWidth + 2 * rootLayout.anchors.margins }
        property int height: { height = rootLayout.implicitHeight + 2 * rootLayout.anchors.margins }
    }
    QQDialogs.FileDialog {
        id: saveFileDialog
        selectMultiple: false
        nameFilters: ["YAML savefile (*.yaml)", "All files (*)"]
        onAccepted: {
            if (selectExisting)
                backend.load(fileUrl)
            else
                backend.save(fileUrl)
        }
    }
    Component.onCompleted: {
        width = settings.width
        height = settings.height
    }
    onClosing: {
        settings.setValue("width", width)
        settings.setValue("height", height)
    }
}
