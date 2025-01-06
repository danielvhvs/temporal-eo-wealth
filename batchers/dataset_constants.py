DHS_COUNTRIES = [
    'angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
    'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
    'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal',
    'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']

LSMS_COUNTRIES = ['ethiopia', 'malawi', 'nigeria', 'tanzania', 'uganda']

SURVEY_NAMES_LSMS = ['ethiopia_2011', 'ethiopia_2015', 'malawi_2010', 'malawi_2016',
                      'nigeria_2010', 'nigeria_2015', 'tanzania_2008', 'tanzania_2012',
                      'uganda_2005', 'uganda_2009', 'uganda_2013']

SIZES = {
    'DHS': {'train': 12319, 'val': 3257, 'test': 4093, 'all': 19669},  # TODO: is this needed?
    'DHS_incountry_A': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_B': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_C': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_D': {'train': 11802, 'val': 3933, 'test': 3934, 'all': 19669},
    'DHS_incountry_E': {'train': 11802, 'val': 3934, 'test': 3933, 'all': 19669},
    'LSMSincountry': {'train': 1812, 'val': 604, 'test': 604, 'all': 3020},  # TODO: is this needed?
}

URBAN_SIZES = {
    'DHS': {'train': 3954, 'val': 1212, 'test': 1635, 'all': 6801},
}

RURAL_SIZES = {
    'DHS': {'train': 8365, 'val': 2045, 'test': 2458, 'all': 12868},
}

# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands

_MEANS_DHS = {
'BLUE': 0.9382496040004136,
 'DMSP': 3.496918903968173,
 'GREEN': 1.0489324988875812,
 'NIR': 1.6570777950268765,
 'RED': 1.1065553613018355,
 'SWIR1': 1.6286681070955282,
 'SWIR2': 1.3444227516200777,
 'VIIRS': 1.2343708792791561
    # 'NIGHTLIGHTS': 5.101585, # nightlights overall
}


_STD_DEVS_DHS = {
'BLUE': 0.0841531335664294,
 'DMSP': 17.03099184867545,
 'GREEN': 0.11621533437193418,
 'NIR': 0.22473480338805057,
 'RED': 0.18743655109705887,
 'SWIR1': 0.32140389259405383,
 'SWIR2': 0.30054984048234507,
 'VIIRS': 5.116964623354029
    # 'NIGHTLIGHTS': 23.342916, # nightlights overall
}


MEANS_DICT = {
    'DHS': _MEANS_DHS,
}

STD_DEVS_DICT = {
    'DHS': _STD_DEVS_DHS,
}
