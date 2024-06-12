import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib

data = pd.read_csv('data/nyc_data.csv')


vic_age_group_valid_values = ['45-64', '18-24', '25-44', '<18', '65+', np.nan]
data = data[data['vic_age_group'].isin(vic_age_group_valid_values)]



# Merge labels
data['offence_description'] = data['offence_description'].replace('PETIT LARCENY OF MOTOR VEHICLE',
                                                                  'LARENCY OF MOTOR VEHICLE')
data['offence_description'] = data['offence_description'].replace('GRAND LARCENY OF MOTOR VEHICLE',
                                                                  'LARENCY OF MOTOR VEHICLE')
data['offence_description'] = data['offence_description'].replace('UNAUTHORIZED USE OF A VEHICLE',
                                                                  'LARENCY OF MOTOR VEHICLE')
data['offence_description'] = data['offence_description'].replace('HARRASSMENT 2', 'HARRASSMENT')
data['offence_description'] = data['offence_description'].replace('ASSAULT 3 & RELATED OFFENSES', 'ASSAULT')
data['offence_description'] = data['offence_description'].replace('FELONY ASSAULT', 'ASSAULT')
data['offence_description'] = data['offence_description'].replace('DANGEROUS DRUGS', 'DRUGS')
data['offence_description'] = data['offence_description'].replace('CANNABIS RELATED OFFENSES', 'DRUGS')
data['offence_description'] = data['offence_description'].replace('UNDER THE INFLUENCE OF DRUGS', 'DRUGS')
data['offence_description'] = data['offence_description'].replace('OFF. AGNST PUB ORD SENSBLTY &',
                                                                  'OFFENCE AGAINST PUBLIC ORDER & SENSIBILITY')
data['offence_description'] = data['offence_description'].replace('DISRUPTION OF A RELIGIOUS SERV',
                                                                  'OFFENCE AGAINST PUBLIC ORDER & SENSIBILITY')
data['offence_description'] = data['offence_description'].replace('DISORDERLY CONDUCT',
                                                                  'OFFENCE AGAINST PUBLIC ORDER & SENSIBILITY')
data['offence_description'] = data['offence_description'].replace('BURGLAR\'S TOOLS', 'BURGLARY')
data['offence_description'] = data['offence_description'].replace('OFFENSES AGAINST PUBLIC ADMINI',
                                                                  'OFFENSES AGAINST PUBLIC SAFETY')
data['offence_description'] = data['offence_description'].replace('OTHER TRAFFIC INFRACTION',
                                                                  'VEHICLE AND TRAFFIC LAWS')
data['offence_description'] = data['offence_description'].replace('INTOXICATED & IMPAIRED DRIVING',
                                                                  'INTOXICATED/IMPAIRED DRIVING')
data['offence_description'] = data['offence_description'].replace('FRAUDULENT ACCOSTING', 'FRAUDS')
data['offence_description'] = data['offence_description'].replace('OFFENSES INVOLVING FRAUD', 'FRAUDS')
data['offence_description'] = data['offence_description'].replace('FRAUDULENT ACCOSTING', 'FRAUDS')
data['offence_description'] = data['offence_description'].replace('OTHER OFFENSES RELATED TO THEF', 'THEFT')
data['offence_description'] = data['offence_description'].replace('THEFT-FRAUD', 'THEFT')
data['offence_description'] = data['offence_description'].replace('THEFT OF SERVICES', 'THEFT')
data['offence_description'] = data['offence_description'].replace('NYS LAWS-UNCLASSIFIED FELONY', 'FELONY')
data['offence_description'] = data['offence_description'].replace('MURDER & NON-NEGL. MANSLAUGHTER', 'FELONY')
data['offence_description'] = data['offence_description'].replace('RAPE', 'FELONY')
data['offence_description'] = data['offence_description'].replace('KIDNAPPING', 'FELONY')
data['offence_description'] = data['offence_description'].replace('ARSON', 'FELONY')
data['offence_description'] = data['offence_description'].replace('LOITERING/DEVIATE SEX', 'SEX CRIMES')
data['offence_description'] = data['offence_description'].replace('FELONY SEX CRIMES', 'SEX CRIMES')
data['offence_description'] = data['offence_description'].replace('PROSTITUTION & RELATED OFFENSES', 'SEX CRIMES')
data['offence_description'] = data['offence_description'].replace('KIDNAPPING AND RELATED OFFENSES',
                                                                  'OFFENSES AGAINST THE PERSON')
data['offence_description'] = data['offence_description'].replace('KIDNAPPING & RELATED OFFENSES',
                                                                  'OFFENSES AGAINST THE PERSON')
data['offence_description'] = data['offence_description'].replace('OFFENSES RELATED TO CHILDREN',
                                                                  'OFFENSES AGAINST THE PERSON')
data['offence_description'] = data['offence_description'].replace('CHILD ABANDONMENT/NON SUPPORT',
                                                                  'OFFENSES AGAINST THE PERSON')
data['offence_description'] = data['offence_description'].replace('OFFENSES AGAINST MARRIAGE UNCL',
                                                                  'OFFENSES AGAINST THE PERSON')
data['offence_description'] = data['offence_description'].replace('KIDNAPPING', 'OFFENSES AGAINST THE PERSON')
data['offence_description'] = data['offence_description'].replace('ANTICIPATORY OFFENSES',
                                                                  'OFFENSES AGAINST THE PERSON')
data['offence_description'] = data['offence_description'].replace('CARDS', 'GAMBLING')
data['offence_description'] = data['offence_description'].replace('LOITERING/GAMBLING', 'GAMBLING')
data['offence_description'] = data['offence_description'].replace('ADMINISTRATIVE CODE', 'ADMINISTRATIVE OFFENSES')
data['offence_description'] = data['offence_description'].replace('ADMINISTRATIVE CODES', 'ADMINISTRATIVE OFFENSES')
data['offence_description'] = data['offence_description'].replace('UNLAWFUL POSS. WEAP. ON SCHOOL', 'DANGEROUS WEAPONS')
data['offence_description'] = data['offence_description'].replace('AGRICULTURE & MRKTS LAW-UNCLASSIFIED', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('NYS LAWS-UNCLASSIFIED VIOLATION', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('NYS LAWS-UNCLASSIFIED FELONY', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('OTHER STATE LAWS (NON PENAL LA', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('OTHER STATE LAWS (NON PENAL LAW)', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('OTHER STATE LAWS', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('MISCELLANEOUS PENAL LAW', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('JOSTLING', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('ENDAN WELFARE INCOMP', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('LOITERING/GAMBLING (CARDS, DIC', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('ESCAPE 3', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('HOMICIDE-NEGLIGENT,UNCLASSIFIE', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('HOMICIDE-NEGLIGENT-VEHICLE', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('NEW YORK CITY HEALTH CODE', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('LOITERING', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('LOITERING FOR DRUG PURPOSES', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('FORTUNE TELLING', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('ABORTION', 'UNKNOWN')
data['offence_description'] = data['offence_description'].replace('ALCOHOLIC BEVERAGE CONTROL LAW', 'UNKNOWN')


# Merge places
data['premises'] = data['premises'].replace('BEAUTY & NAIL SALON', 'BEAUTY/NAIL SALON')
data['premises'] = data['premises'].replace('CHECK CASHING BUSINESS', 'CHECK CASH')
data['premises'] = data['premises'].replace('CLOTHING/BOUTIQUE', 'CLOTHING BOUTIQUE')
data['premises'] = data['premises'].replace('COMMERCIAL BLDG', 'COMMERCIAL BUILDING')
data['premises'] = data['premises'].replace('DEPARTMENT STORE', 'DEPT STORE')
data['premises'] = data['premises'].replace('DOCTOR/DENTIST OFFICE', 'DOCTOR/DENTIST')
data['premises'] = data['premises'].replace('FOOD SUPERMARKET', 'SUPERMARKET')
data['premises'] = data['premises'].replace('JEWELRY STORE', 'JEWELRY')
data['premises'] = data['premises'].replace('SHOE STORE', 'SHOE')
data['premises'] = data['premises'].replace('SOCIAL CLUB/POLICY LOCATI', 'SOCIAL CLUB/POLICY')

sample = data.__deepcopy__()


def convert_time_to_minutes(row):
    try:
        temp = row.split(':')
        return int(temp[0]) * 60 + int(temp[1])
    except ValueError:
        return None
    except AttributeError:
        return None


# Encode data
sample.loc[:, 'time_of_call'] = sample.time_of_call.apply(convert_time_to_minutes)

time_normalizer = MinMaxScaler().fit(sample[['time_of_call']])
sample.loc[:, 'time_of_call'] = time_normalizer.transform(sample[['time_of_call']])

vic_age_encoder = LabelEncoder().fit(sample.vic_age_group)
sample.loc[:, 'vic_age_group'] = vic_age_encoder.transform(sample.vic_age_group)

vic_race_encoder = LabelEncoder().fit(sample.vic_race)
sample.loc[:, 'vic_race'] = vic_race_encoder.transform(sample.vic_race)

vic_sex_encoder = LabelEncoder().fit(sample.vic_sex)
sample.loc[:, 'vic_sex'] = vic_sex_encoder.transform(sample.vic_sex)

suspect_race_encoder = LabelEncoder().fit(sample.suspect_race)
sample.loc[:, 'suspect_race'] = suspect_race_encoder.transform(sample.suspect_race)

suspect_sex_encoder = LabelEncoder().fit(sample.suspect_sex)
sample.loc[:, 'suspect_sex'] = suspect_sex_encoder.transform(sample.suspect_sex)

precinct_encoder = LabelEncoder().fit(sample.precinct)
sample.loc[:, 'precinct'] = precinct_encoder.transform(sample.precinct)

borough_encoder = LabelEncoder().fit(sample.borough)
sample.loc[:, 'borough'] = borough_encoder.transform(sample.borough)

location_of_occurrence_encoder = LabelEncoder().fit(sample.location_of_occurrence)
sample.loc[:, 'location_of_occurrence'] = location_of_occurrence_encoder.transform(
    sample.location_of_occurrence)

premises_encoder = LabelEncoder().fit(sample.premises)
sample.loc[:, 'premises'] = premises_encoder.transform(sample.premises)

offence_description_encoder = LabelEncoder().fit(sample.offence_description)
sample.loc[:, 'offence_description'] = offence_description_encoder.transform(sample.offence_description)


def sample_func(group):
    if len(group) > 50000:
        return group.sample(50000)
    else:
        return group


def save_filtered_data(df, filename):
    # Group by category and count occurrences
    counts = df['offence_description'].value_counts()

    # Filter out categories with less than 1000 occurrences
    df = df[df['offence_description'].isin(counts[counts > 1000].index)]

    df = df.groupby('offence_description').apply(sample_func).reset_index(drop=True)
    df.to_csv(f'data/{filename}.csv', index=False)


def save_encoders():
    encoders = {'time_of_call': time_normalizer, 'vic_age_group': vic_age_encoder, 'vic_race': vic_race_encoder,
                'vic_sex': vic_sex_encoder,
                'suspect_race': suspect_race_encoder, 'suspect_sex': suspect_sex_encoder, 'precinct': precinct_encoder,
                'borough': borough_encoder, 'location_of_occurrence': location_of_occurrence_encoder, 'premises': premises_encoder,
                'offence_description': offence_description_encoder}

    for name, encoder in encoders.items():
        joblib.dump(encoder, f'encoders/{name}.pkl')



def save_train_test_split_data():
    X = sample.drop(columns=['offence_description'])
    y = sample['offence_description']
    y = y.astype('int')
    X = X.astype('float')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    joblib.dump(X_train, 'data/X_train.pkl')
    joblib.dump(X_test, 'data/X_test.pkl')
    joblib.dump(y_train, 'data/y_train.pkl')
    joblib.dump(y_test, 'data/y_test.pkl')


# Save filtered data, with no encoding applied
save_filtered_data(data, 'selected_data')
# Save filtered data, with encoding applied
save_filtered_data(sample, 'encoded_selected_data')
# Save encoders
save_encoders()
# Save train_test data
save_train_test_split_data()
