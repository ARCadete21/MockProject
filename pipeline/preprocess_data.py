import json
import pandas as pd
import numpy as np
from .utils import (
    calculate_mean_difference,
    get_years_of_education,
    drop_missing_values,
    transform_variables_to_boolean
)


class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data


    def _handle_inconsistencies(self):
        evening_courses = self.data['Course'].str.contains('evening')
        morning_participations = (self.data['Morning shift participation'] == True)
        self.data.loc[evening_courses & morning_participations, 'Morning shift participation'] = np.nan

        period1 = self.data.copy().iloc[:, 21:27]
        period2 = self.data.copy().iloc[:, 27:33]

        approved_credited1 = period1['N units approved 1st period'] < period1['N units credited 1st period']
        approved_credited2 = period2['N units approved 2nd period'] < period2['N units credited 2nd period']
        self.data.loc[approved_credited1, ['N units approved 1st period', 'N units credited 1st period']] = np.nan
        self.data.loc[approved_credited2, ['N units approved 2nd period', 'N units credited 2nd period']] = np.nan

        taken1 = (period1['N units taken 1st period'] > 0) & (period1['N scored units 1st period'] == 0) & (period1['N unscored units 1st period'] == 0)
        taken2 = (period2['N units taken 2nd period'] > 0) & (period2['N scored units 2nd period'] == 0) & (period2['N unscored units 2nd period'] == 0)
        self.data.loc[taken1, 'N units taken 1st period'] = 0
        self.data.loc[taken2, 'N units taken 2nd period'] = 0

        self.data.drop(columns=['Registered'], inplace=True)
        self.data['Average grade 1st period'] = self.data['Average grade 1st period'] * 10
        self.data['Average grade 2nd period'] = self.data['Average grade 2nd period'] * 10


    def _handle_categorcial_data(self):
        self.data['Course'] = self.data['Course'].replace('Equinculture', 'Echinculture')
        self.data['Course'] = self.data['Course'].str.replace(' (evening attendance)', '')
        self.data['Course'] = self.data['Course'].str.capitalize()

        self.data = calculate_mean_difference(self.data, 'Entry score', ['Course', 'Application mode'], 'Course application mode entry score difference')[0]

        for col in [f'Average grade {semester} period' for semester in ['1st', '2nd']]:
            self.data = calculate_mean_difference(self.data, col, ['Course'], f'Course {col.lower()} difference')[0]

        self.data['Marital status'] = self.data['Marital status'].apply(lambda x: True if x == 'single' else False)
        self.data['Nationality'] = self.data['Nationality'].apply(lambda x: True if x == 'Portuguese' else False)

        for parent in ['Mother', 'Father']:
            var = f"{parent}'s occupation"
            self.data[var] = self.data[var].apply(lambda x: False if x == 'Unskilled Worker' else True)

        self.data['Technological course'] = self.data['Previous qualification'].astype(str).apply(lambda x: True if 'Technological' in x else False)

        with open('pipeline/support/apriori/areas_dict.json', 'r') as json_file:
            areas_dict = json.load(json_file)

        self.data['Course area'] = self.data['Course'].apply(self.map_course_to_area, args=(areas_dict,))
        self.data.drop(columns=['Course'], inplace=True)

        for person in ['Previous', 'Mother\'s', 'Father\'s']:
            var = f'{person} qualification'
            self.data[var] = self.data[var].astype(str).apply(get_years_of_education)


    def preprocessor(self):
        self.data.drop_duplicates(inplace=True)
        self.data = drop_missing_values(1, self.data)[0]
        self.data = drop_missing_values(0, self.data)[0]
        self.data = transform_variables_to_boolean(self.data)
        self._handle_inconsistencies()
        self._handle_categorcial_data()
        return self.data


    @staticmethod
    def map_course_to_area(course, areas_dict):
        words = course.split()
        for area, courses in areas_dict.items():
            if course in courses or (words and words[-1].capitalize() in courses):
                return area
        return 'Engineering and related techniques'