// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Dialogs 1.3 as QQDialogs
import QtQuick.Layouts 1.12
import Qt.labs.settings 1.0
import SdtGui 0.1 as Sdt
import FRETInspector 1.0


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
                        saveFileDialog.selectExisting = true
                        saveFileDialog.open()
                    }
                }
                ToolButton {
                    icon.name: "document-save"
                    onClicked: {
                        // saveFileDialog.selectExisting = false
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
                        target: fileSel.currentModelData.particles
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
                        target: fileSel.currentModelData.particles
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
                        target: fileSel.currentModelData.particles
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
                            backend.plot(particleSel.currentModelData.smData, checked)
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
                            frameSel.value = backend.firstFrame(
                                particleSel.currentModelData.dTrackData)
                        }
                    }
                    ToolButton {
                        icon.name: "go-last"
                        Layout.fillWidth: true
                        onClicked: {
                            frameSel.value = backend.lastFrame(
                                particleSel.currentModelData.dTrackData)
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
                            fileSel.currentModelData.particles.manuallyFilterTrack(
                                particleSel.currentIndex, false)
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
                            fileSel.currentModelData.particles.manuallyFilterTrack(
                                particleSel.currentIndex, true)
                            if (goNext)
                                particleSelGroup.nextParticle()
                        }
                    }
                    Label {
                        Layout.columnSpan: 2
                        Layout.alignment: Qt.AlignCenter
                        text: "filter: " + filterText(particleSel.currentModelData.manualFilter)

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
                Sdt.ModelComboBox {
                    id: fileSel

                    model: datasetSel.currentDataset
                    textRole: "display"
                    modelDataRole: ["particles", "ddImg", "daImg", "aaImg"]
                    selectFirstOnReset: true
                    Layout.fillWidth: true
                }
                Item { width: 5 }
                Label { text: "particle" }
                Sdt.ModelComboBox {
                    id: particleSel

                    model: fileSel.currentModelData.particles
                    textRole: "display"
                    modelDataRole: ["number", "smData", "dTrackData", "aTrackData", "manualFilter"]
                    onCurrentModelDataChanged: {
                        backend.plot(currentModelData.smData, scatterCheck.checked)
                        if (autoFirstCheck.checked)
                            frameSel.value = backend.firstFrame(currentModelData.dTrackData)
                    }
                }
                Item { width: 5 }
                Label { text: "frame" }
                Sdt.EditableSpinBox {
                    id: frameSel
                    to: backend.frameCount(fileSel.currentModelData)
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
                    image: backend.image(fileSel.currentModelData.ddImg,
                                         frameSel.value)
                    overlays: Sdt.TrackDisplay {
                        trackData: particleSel.currentModelData.dTrackData
                        currentFrame: frameSel.value
                        visible: showLocCheck.checked
                    }
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                Sdt.ImageDisplay {
                    image: backend.image(fileSel.currentModelData.daImg,
                                         frameSel.value)
                    overlays: Sdt.TrackDisplay {
                        trackData: particleSel.currentModelData.aTrackData
                        currentFrame: frameSel.value
                        visible: showLocCheck.checked
                    }
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                Sdt.ImageDisplay {
                    image: backend.image(fileSel.currentModelData.aaImg,
                                         frameSel.value)
                    overlays: Sdt.TrackDisplay {
                        trackData: particleSel.currentModelData.aTrackData
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
