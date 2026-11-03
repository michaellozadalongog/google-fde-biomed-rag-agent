# Ultrasound Tissue-Mimicking Phantom QA Protocol (Synthetic, v1.2)

## Purpose
This protocol defines weekly quality assurance (QA) checks for B-mode ultrasound systems using a tissue-mimicking phantom. It is a synthetic teaching document written for this project and is not a regulatory document.

## Speed of sound
Tissue-mimicking material should have a speed of sound of 1540 m/s ± 10 m/s at 22 °C. Scanners assume 1540 m/s when converting echo time to depth, so a phantom outside this tolerance produces depth calibration errors.

## Attenuation
Target attenuation coefficient is 0.5 dB/cm/MHz ± 0.05. Higher attenuation reduces penetration depth; record maximum depth of penetration (DOP) at a fixed gain and compare to baseline. A DOP drop greater than 1 cm from baseline triggers service review.

## Axial and lateral resolution
Use the nylon filament groups (0.1 mm diameter) spaced 0.25, 0.5, 1.0 and 2.0 mm. Axial resolution is the smallest axial spacing at which two filaments are displayed as separate echoes. For a 7.5 MHz linear probe, expected axial resolution is at or below 0.5 mm. Lateral resolution degrades outside the focal zone; record it at the focal depth.

## Dead zone
The dead zone (ring-down) is the depth from the transducer face where echoes cannot be resolved. Acceptable dead zone for linear arrays is less than 3 mm.

## Uniformity and dropouts
Image the uniform region. Vertical dark bands indicate dead or weak transducer elements. Any persistent vertical dropout fails the test and the probe must be removed from clinical service.

## Documentation
Record probe serial number, preset, gain, depth, focal zone, and operator initials. Store images with the QA log.
