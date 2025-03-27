"""
Module responsible for CT image acquisition simulation
"""

from pathlib import Path
from shutil import rmtree
from datetime import datetime
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import gecatsim as xc
from monai.data import MetaTensor

from gecatsim.reconstruction.pyfiles import recon
from .ground_truth_definition.phantoms import (voxelize_ground_truth,
                                               Phantom)

install_path = Path(__file__).parent


def read_dicom(dcm_fname: str) -> np.ndarray:
    '''
    Reads dicom file and returns numpy array

    :param dcm_fname: dicom filename to be read
    '''
    dcm = pydicom.dcmread(dcm_fname)
    return dcm.pixel_array + int(dcm.RescaleIntercept)


def convert_to_dicom(img_slice: np.ndarray, phantom_path: str,
                     spacings: tuple):
    '''
    :param img_slice: input 2D ndarray to be saved
    :param phantom_path: filename to save dicom file to
    :param spacings: tuple containing pixel spacings in mm
    '''
    # https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144
    Path(phantom_path).parent.mkdir(exist_ok=True, parents=True)
    fpath = pydicom.data.get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(fpath)
    ds.Rows, ds.Columns = img_slice.shape
    ds.SliceThickness = spacings[0]
    ds.PixelSpacing = [spacings[1], spacings[2]]
    ds.PixelData = (img_slice.copy(order='C').astype('int16') -\
        int(ds.RescaleIntercept)).tobytes()
    pydicom.dcmwrite(phantom_path, ds)


def get_projection_data(ct):
    '''takes as input xcist cfg struct and returns ndarray'''
    return xc.rawread(ct.resultsName+'.prep',
                      [ct.protocol.viewCount,
                       ct.scanner.detectorRowCount,
                       ct.scanner.detectorColCount],
                      'float')


def get_reconstructed_data(ct) -> np.ndarray:
    '''takes as input xcist cfg struct and returns ndarray'''
    imsize = ct.recon.imageSize
    return xc.rawread(
        ct.resultsName+f'_{imsize}x{imsize}x{ct.recon.sliceCount}.raw',
        [ct.recon.sliceCount, imsize, imsize], 'single')


def initialize_xcist(ground_truth_image, spacings=(1, 1, 1),
                     scanner_model='Scanner_Default',
                     output_dir='default', phantom_id='default',
                     materials=None):
    '''
    :param fov: in mm
    :param spacings: z, x, y in mm
    '''
    print('Initializing Scanner object...')
    print(''.join(10*['-']))
    # load defaults
    scanner_path=install_path/'defaults/'/scanner_model

    ct = xc.CatSim(install_path/'defaults/Phantom_Default',
                   install_path/'defaults/Physics_Default',
                   install_path/'defaults/Protocol_Default',
                   install_path/'defaults/Recon_Default',
                   scanner_path)

    ct.cfg.waitForKeypress = False
    ct.cfg.do_Recon = True

    # prepare directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    phantom_path = output_dir / 'phantoms' / f'{phantom_id}'
    phantom_path.mkdir(exist_ok=True, parents=True)
    ct.cfg.phantom.filename = str(phantom_path / f'{phantom_id}.json')

    # prepare material density arrays from ground truth phantom
    if ground_truth_image.ndim == 2:
        ground_truth_image = ground_truth_image[None]

    dicom_path = phantom_path / 'dicom'
    for slice_id, img in enumerate(ground_truth_image):
        dicom_filename = dicom_path / f'1-{slice_id:03d}.dcm'
        convert_to_dicom(img, dicom_filename, spacings=spacings)

    voxelize_ground_truth(dicom_path, phantom_path,
                          material_threshold_dict=materials)
    print('Scanner Ready')
    return ct


_table_speed = {'Low': 26.67, 'Intermediate': 48, 'High': 64}
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5711061/


class Scanner():
    """
    A class to hold CT simulation data and run simulations
    """
    def __init__(self, phantom: Phantom, scanner_model: str = "Scanner_Default",
                 studyname: str = "default",
                 studyid: int = 0, seriesname: str = "default", seriesid=0,
                 framework: str = 'CATSIM', output_dir: str | Path = None,
                 materials: dict | None = None) -> None:
        """
        :param phantom: Phantom class instance to be scanned, voxels in units of
            approximate CT Numbers [HU], typically in python
            coordinates (z, x, y)
            where z is perpendicular to the axial plane made by x and y.
            See <https://en.wikipedia.org/wiki/Hounsfield_scale>
            for some suggested values for common materials

        :param scanner_model: str, study identifier to be used for virtual identifier and DICOM header

        :param studyname: str, study identifier to be saved in DICOM header
        :param studyid: int, study identifier to be saved in DICOM header
        :param seriesname: str, series identifier to be saved in DICOM header
        :param seriesid: int, series identifier to be saved in DICOM header
        :param output_dir: optional directory to save the intermediate and
            final simulation results, defaults to current working directory
        :param framework: Optional, CT simulation framework options
            include `['CATSIM']`
        :param materials: Optional dictionary of {material name: HU value},
            used for construction volume fraction maps in XCIST,
            see materials and thresholds from this XCIST example:
            https://github.com/xcist/phantoms-voxelized/blob/main/DICOM_to_voxelized/DICOM_to_voxelized_example_head.cfg

        :returns: None

        See also <https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L19>
        """
        output_dir = output_dir or '.'
        output_dir = Path(output_dir) / f'{phantom.patient_name}'
        if output_dir.exists():
            rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

        img = phantom.get_CT_number_phantom()
        if isinstance(img, MetaTensor):
            img = img.numpy()

        self.phantom = phantom
        available_scanners = [o.name for o in install_path.glob('defaults/*')
                              if not str(o).endswith('.cfg')]
        if scanner_model not in available_scanners:
            raise ValueError(f'{scanner_model} not in {available_scanners}')
        self.scanner_model = scanner_model
        self.studyname = studyname or self.patientname
        self.studyid = studyid
        self.seriesname = seriesname
        self.seriesid = seriesid
        self.framework = framework
        self.ndetectors = 900
        self.nangles = 1000
        self.detector_size = 1
        self.recon = None
        self.projections = None
        self.groundtruth = None
        self.patient_diameter = 18
        self.zspan = 'dynamic'

        self.xcist = initialize_xcist(img, self.phantom.spacings,
                                      scanner_model=self.scanner_model,
                                      output_dir=self.output_dir,
                                      phantom_id=phantom.patientid,
                                      materials=materials)
        self.start_positions = self.calculate_start_positions()

    def calculate_start_positions(self):
        'determine number of axial scans required to cover the phantom'
        detector_width = self.xcist.scanner.detectorRowCount *\
            self.xcist.scanner.detectorRowSize
        magnification = self.xcist.scanner.sdd / self.xcist.scanner.sid
        detector_width_at_isocenter = detector_width / magnification

        safe_width_at_isocenter = detector_width_at_isocenter -\
            2*self.xcist.scanner.detectorRowSize
        self.scan_width = self.xcist.cfg.protocol.rotationTime *\
            self.xcist.cfg.protocol.tableSpeed + safe_width_at_isocenter
        # img = self.phantom.get_CT_number_phantom()
        self.total_scan_length = self.phantom.spacings[0]*self.phantom.shape[0]
        return np.arange(-self.total_scan_length/2,
                         self.total_scan_length/2,
                         self.scan_width)

    def recommend_scan_range(self, threshold=-950) -> tuple:
        '''
        returns recommended startZ and endZ based on scout scan
        attenuation profile

        threshold [HU] determines minimum attenuating regions to keep
        '''
        img = self.phantom.get_CT_number_phantom()
        scout_profile = img.mean(axis=(1, 2))
        suggested_start_idx = np.argmax(np.diff(scout_profile > threshold)) + 1
        suggested_start_mm = self.start_positions[0] + suggested_start_idx *\
            self.phantom.spacings[0]

        if np.all(scout_profile[suggested_start_idx:] > threshold):
            suggested_end_mm = self.start_positions[-1]
        else:
            suggested_end_idx = \
                np.argmax(np.diff(scout_profile[suggested_start_idx:] >
                                  threshold)) + 1
            suggested_end_mm = self.start_positions[0] + suggested_end_idx *\
                self.phantom.spacings[0]
        return (int(suggested_start_mm), int(suggested_end_mm))

    def get_lesion_mask(self, startZ: int | None = None,
                        endZ: int | None = None,
                        fov: float | None = None,
                        slice_thickness=1, **kwargs) -> np.ndarray[bool]:
        '''takes lesion in object space and returns a mask in CT image space
        for the given imaging system'''
        if not self.phantom._lesion:
            return
        ground_truth_lesion = self.phantom._lesion[0]
        lesion_phantom = deepcopy(self.phantom)
        lesion_phantom._phantom = np.where(ground_truth_lesion > 0, 0, - 1000)
        lesion_phantom.patient_name = 'lesion only'
        lesion_dir = self.output_dir / 'lesion_mask'
        lesion_only = Scanner(lesion_phantom,
                              scanner_model=self.scanner_model,
                              materials={
                                'ICRU_lung_adult_healthy': -1000,
                                'water': -100},
                              output_dir=lesion_dir)
        lesion_only.xcist.cfg.physics.energyCount = 2
        lesion_only.xcist.cfg.physics.monochromatic = -1
        lesion_only.xcist.cfg.physics.enableElectronicNoise = 0
        lesion_only.xcist.cfg.physics.enableQuantumNoise = 0
        lesion_only.run_scan(mA=500, views=100, startZ=startZ, endZ=endZ)
        lesion_only.run_recon(sliceThickness=slice_thickness, fov=fov)
        rmtree(lesion_dir)
        return (lesion_only.recon > -950) & (self.recon > -300)

    def scout_view(self, startZ=None, endZ=None, table_speed=0):
        '''
        Preview radiograph useful for determining scan range startZ and endZ
        :param startZ: optional starting table position in mm of the scan,
            see self.start_positions
        :param endZ: optional last position of scan in mm,
            see self.start_positions
        '''

        if isinstance(table_speed, str):
            self.xcist.cfg.protocol.tableSpeed = _table_speed[table_speed]
        else:
            self.xcist.cfg.protocol.tableSpeed = table_speed

        self.start_positions = self.calculate_start_positions()
        start_positions = self.start_positions.copy()
        if startZ is not None:
            if startZ < start_positions.min():
                raise ValueError(f'startZ is outside the range of valid\
                                  start positions: {self.start_positions}')
            start_positions = start_positions[start_positions > startZ]
        if endZ is not None:
            if endZ > self.start_positions.max():
                raise ValueError(f'startZ is outside the range of valid\
                                  start positions: {self.start_positions}')
            start_positions = start_positions[start_positions < endZ]
        img = self.phantom.get_CT_number_phantom()
        plt.imshow(img.sum(axis=1), cmap='gray', origin='lower',
                   extent=[-img.shape[0]*self.phantom.spacings[0]/2,
                           img.shape[0]*self.phantom.spacings[0]/2,
                           self.start_positions[0]+self.total_scan_length,
                           self.start_positions[0]])
        plt.hlines(y=start_positions[0],
                   xmin=-img.shape[0]*self.phantom.spacings[0] / 2,
                   xmax=img.shape[0]*self.phantom.spacings[0]/2, color='red')
        plt.annotate('Stop', (0, start_positions[0]-10),
                     horizontalalignment='center')

        plt.hlines(y=start_positions[-1] + self.scan_width,
                   xmin=-img.shape[0]*self.phantom.spacings[0]/2,
                   xmax=img.shape[0]*self.phantom.spacings[0]/2, color='red')
        plt.annotate('Start', (0, start_positions[-1] + self.scan_width + 10),
                     horizontalalignment='center')

        plt.annotate(f'{len(start_positions)} scans required',
                     xy=(0, (start_positions[0]+start_positions[-1])/2),
                     horizontalalignment='center')
        plt.annotate('', xy=(40, start_positions[-1] + self.scan_width),
                     xytext=(40, start_positions[0]),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.title(f'Table Speed: {self.xcist.cfg.protocol.tableSpeed} mm/s')
        plt.ylabel('scan z position [mm]')
        plt.xlabel('scan x position [mm]')

    def __repr__(self) -> str:
        repr = f'''
        {self.__class__} {self.seriesname}
        Scanner: {self.scanner_model}
        Simulation Platform: {self.framework}
        '''
        if self.recon is None:
            return repr
        repr += f'\nRecon: {self.recon.shape} {self.xcist.cfg.recon.fov/10} cm fov'
        if self.projections is None:
            return repr
        repr += f'\nProjections: {self.projections.shape}'
        return repr

    def run_scan(self, mA=200, kVp=120, startZ=None, endZ=None, views=None,
                 table_speed=0, bhc=True):
        """
        Runs the CT simulation using the stored parameters.

        :param mA: x-ray source milliamps, increases x-ray flux linearly,
            $noise ~ 1/sqrt(mA)$
        :param kVp: x-ray source potential, increases x-ray flux
            nonlinearly and reduces contrast as increased
        :param startZ: optional starting table position in mm of the scan,
            see self.start_positions
        :param endZ: optional last position of scan in mm,
            see self.start_positions
        :param views: number of angular views, for testing this can be
            reduced but will produced aliasing streaks
        :param verbose: optional boolean, if True prints out status
            updates, if False they are suppressed.
        :param table_speed: optional [float, str] str options include
            'Low': 26.67, 'Intermediate': 48, 'High': 64, units in mm/s
        :param bhc: optional [bool, str], if True applies polynomial beam
            hardening correction to correct for polychromatic cupping artifact,
            if `default` applied default XCIST bhc, caution this gives capping
            artifacts. Options include [True, False, 'default']
        """
        self.xcist.cfg.protocol.mA = mA
        kVp_options = [70, 80, 90, 100, 110, 120, 130, 140]
        if kVp not in kVp_options:
            raise ValueError(f'Selected kVP [{kVp}] not available,\
                              please choose from {kVp_options}')
        self.xcist.cfg.protocol.spectrumFilename =\
            f'tungsten_tar7.0_{kVp}_filt.dat'
        self.kVp = kVp
        if isinstance(table_speed, str):
            self.xcist.cfg.protocol.tableSpeed = _table_speed[table_speed]
        else:
            self.xcist.cfg.protocol.tableSpeed = table_speed

        self.start_positions = self.calculate_start_positions()
        start_positions = self.start_positions

        if startZ is not None:
            if startZ < start_positions.min():
                raise ValueError(f'startZ is outside the range of valid\
                                  start positions: {self.start_positions}')
            start_positions = start_positions[start_positions > startZ]
        if endZ is not None:
            if endZ > start_positions.max():
                raise ValueError(f'startZ is outside the range of valid\
                                  start positions: {self.start_positions}')
            start_positions = start_positions[start_positions < endZ]

        if views:
            self.xcist.cfg.protocol.viewCount = views
            self.xcist.protocol.stopViewId =\
                self.xcist.cfg.protocol.startViewId +\
                self.xcist.cfg.protocol.viewCount-1
            self.xcist.cfg.protocol.viewsPerRotation = views

        if bhc is True:
            self.xcist.cfg.physics.callback_post_log = 'Prep_BHC_Accurate'
            self.xcist.cfg.physics.EffectiveMu = 0.2
            self.xcist.cfg.physics.BHC_poly_order = 5
            self.xcist.cfg.physics.BHC_max_length_mm = int(self.phantom.size[1])
            self.xcist.cfg.physics.BHC_length_step_mm = 10
        elif bhc is False:
            self.xcist.cfg.physics.callback_post_log = ""
            self.xcist.cfg.protocol.bowtie = ""

        self.results_dir = self.output_dir / 'simulations' / \
            f'{self.phantom.patientid}'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.xcist.protocol.spectrumFilename = f"tungsten_tar7.0_{int(kVp)}_filt.dat"  # name of the spectrum file
        self.xcist.cfg.experimentDirectory = str(self.results_dir)

        projections = []
        for idx, table_position in enumerate(start_positions):
            print(f'scan: {idx+1}/{len(start_positions)}')
            self.xcist.cfg.resultsName = str((self.results_dir / f'{idx:03d}_{mA}mA_{kVp}kV').absolute())  # keep projection data from each scan
            self.xcist.resultsName = self.xcist.cfg.resultsName
            self.xcist.protocol.startZ = table_position
            self.xcist.run_all()
            projections.append(self.xcist.cfg.resultsName)
        self._projections = projections
        return self

    def run_recon(self, fov=None, sliceThickness=None, sliceCount=None,
                  mu_water=None, kernel='standard'):
        '''
        perform reconstruction and save to .recon attribute
        :param kernel: reconstruction kernel, options include: ['standard',
            'soft', 'bone', 'R-L', 'S-L']
            'R-L': Ramachandran-Lakshminarayanan (R-L) filter
            'S-L' for Shepp-Logan (S-L) filter
            See: https://github.com/xcist/main/blob/master/gecatsim/cfg/Recon_Default.cfg
        '''
        defined_kernels = ['standard', 'soft', 'bone', 'R-L', 'S-L']
        if kernel not in defined_kernels:
            raise ValueError(f'{kernel} not in {defined_kernels}')
        else:
            self.xcist.cfg.recon.kernelType = kernel
        if sliceThickness:
            self.xcist.recon.sliceThickness = sliceThickness
        if mu_water:
            self.xcist.cfg.recon.mu = mu_water
        else:
            if self.xcist.physics.monochromatic != -1:
                self.xcist.recon.mu = xc.GetMu(
                    'water', self.xcist.physics.monochromatic)[0]/10
        if not sliceCount:
            detector_width = self.xcist.scanner.detectorRowCount *\
                self.xcist.scanner.detectorRowSize
            magnification = self.xcist.scanner.sdd / self.xcist.scanner.sid
            detector_width_at_isocenter = detector_width / magnification
            safe_width_at_isocenter = detector_width_at_isocenter -\
                2*self.xcist.scanner.detectorRowSize
            valid_slices = int(safe_width_at_isocenter //
                               self.xcist.recon.sliceThickness)
            self.xcist.cfg.recon.sliceCount = valid_slices
        else:
            self.xcist.cfg.recon.sliceCount = sliceCount
        if fov:
            self.xcist.cfg.recon.fov = fov
        print(f'fov size: {self.xcist.cfg.recon.fov}')

        self.xcist.cfg.recon.displayImagePictures = False

        recons = []
        for proj in self._projections:
            self.xcist.cfg.resultsName = proj
            self.xcist.resultsName = self.xcist.cfg.resultsName
            vol = recon.recon_direct(self.xcist.cfg).transpose(2, 0, 1)
            recons.append(vol)
        self.recon = np.concatenate(recons, axis=0)
        self.projections = get_projection_data(self.xcist)
        self.groundtruth = None
        self.I0 = self.xcist.cfg.protocol.mA
        self.nsims = 1
        return self

    def write_to_dicom(self, fname: str | Path,
                       groundtruth=False) -> list[Path]:
        """
        write ct data to DICOM file, returns list of written dicom file names

        :param fname: filename to save image to (preferably with '.dcm`
            or related extension)
        :param groundtruth: Optional, whether to save the ground truth
            phantom image (no noise, blurring, or other artifacts).
            If True, 'self.groundtruth` is saved, if False (default)
            `self.recon` which contains blurring (and noise if 'add_noise`True)
        :returns: list[Path]

        Adapted from <https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144>
        """
        fpath = pydicom.data.get_testdata_file("CT_small.dcm")
        ds = pydicom.dcmread(fpath)
        # update meta info
        ds.Manufacturer = 'GE (simulated)'
        ds.ManufacturerModelName = 'LightSpeed 16 (simulated)'
        time = datetime.now()
        ds.InstanceCreationDate = time.strftime('%Y%m%d')
        ds.InstanceCreationTime = time.strftime('%H%M%S')
        ds.InstitutionName = 'FDA/CDRH/OSEL/DIDSR'
        ds.StudyDate = ds.InstanceCreationDate
        ds.StudyTime = ds.InstanceCreationTime
        ds.PatientName = self.phantom.patient_name
        ds.SeriesNumber = self.seriesid
        ds.PatientAge = f'{int(self.phantom.age):03d}Y'
        ds.PatientID = f'{int(self.phantom.patientid):03d}'
        del ds.PatientWeight
        del ds.ContrastBolusRoute
        del ds.ContrastBolusAgent
        ds.ImageComments =\
            f"effctive diameter [cm]: {self.patient_diameter/10}"
        ds.ScanOptions = self.xcist.cfg.protocol.scanTrajectory.upper()
        ds.ReconstructionDiameter = self.xcist.cfg.recon.fov
        ds.ConvolutionKernel = self.xcist.cfg.recon.kernelType
        ds.Exposure = self.xcist.cfg.protocol.mA

        # load image data
        ds.StudyDescription = f"{self.I0} photons " + self.seriesname +\
            " " + ds.ConvolutionKernel + self.xcist.cfg.recon.reconType
        if self.recon.ndim == 2:
            self.recon = self.recon[None]
        nslices, ds.Rows, ds.Columns = self.recon.shape

        ds.SpacingBetweenSlices = ds.SliceThickness
        ds.DistanceSourceToDetector = self.xcist.cfg.scanner.sdd
        ds.DistanceSourceToPatient = self.xcist.cfg.scanner.sid

        ds.PixelSpacing = [self.xcist.cfg.recon.fov/self.xcist.cfg.recon.imageSize,
                           self.xcist.cfg.recon.fov/self.xcist.cfg.recon.imageSize]
        ds.SliceThickness = self.xcist.cfg.recon.sliceThickness

        ds.KVP = self.kVp
        ds.StudyID = str(self.studyid)
        # series instance uid unique for each series
        end = ds.SeriesInstanceUID.split('.')[-1]
        new_end = str(int(end) + self.studyid)
        ds.SeriesInstanceUID = ds.SeriesInstanceUID.replace(end, new_end)

        # study instance uid unique for each series
        end = ds.StudyInstanceUID.split('.')[-1]
        new_end = str(int(end) + self.studyid)
        ds.StudyInstanceUID = ds.StudyInstanceUID.replace(end, new_end)
        ds.AcquisitionNumber = self.studyid

        fname = Path(fname)
        fname.parent.mkdir(exist_ok=True, parents=True)
        # saveout slices as individual dicom files
        fnames = []
        vol = self.groundtruth if groundtruth else self.recon
        if vol.ndim == 2:
            vol = vol[None]
        for slice_idx, array_slice in enumerate(vol):
            ds.InstanceNumber = slice_idx + 1  # image number
            # SOP instance UID changes every slice
            end = ds.SOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + self.studyid + self.seriesid)
            ds.SOPInstanceUID = ds.SOPInstanceUID.replace(end, new_end)
            # MediaStorageSOPInstanceUID changes every slice
            end = ds.file_meta.MediaStorageSOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + self.studyid + self.seriesid)
            ds.file_meta.MediaStorageSOPInstanceUID =\
                ds.file_meta.MediaStorageSOPInstanceUID.replace(end, new_end)
            # slice location and image position changes every slice
            ds.SliceLocation = self.nsims//2*ds.SliceThickness +\
                slice_idx*ds.SliceThickness
            ds.ImagePositionPatient[-1] = ds.SliceLocation
            ds.ImagePositionPatient[0] = -ds.Rows//2*ds.PixelSpacing[0]
            ds.ImagePositionPatient[1] = -ds.Columns//2*ds.PixelSpacing[1]
            ds.ImagePositionPatient[2] = ds.SliceLocation
            ds.PixelData = (array_slice.copy(order='C').astype('int16') -\
                int(ds.RescaleIntercept)).tobytes()
            dcm_fname = fname.parent /\
                f'{fname.stem}_{slice_idx:03d}{fname.suffix}'\
                if nslices > 1 else fname
            fnames.append(dcm_fname)
            pydicom.dcmwrite(dcm_fname, ds)
        return fnames
