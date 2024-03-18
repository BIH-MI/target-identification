from enum import Enum

import yaml


class ConfigAttribute:
    """
    Class to represent a single attribute's configuration.
    """

    def __init__(self, name, attr_type, data_type, order=None, levels=None, path_to_hierarchy=None):
        self.name = name
        self.type = attr_type
        self.data_type = data_type
        self.order = order
        self.levels = levels
        self.path_to_hierarchy = path_to_hierarchy


class ATTRIBUTES(Enum):
    ALL = 1
    QI_ONLY = 2
    LIST = 3


class DatasetConfiguration:
    """
    Class to represent a dataset configuration.
    """

    def __init__(self, config_file, target_attributes=ATTRIBUTES.ALL):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.data_set_name = self.config['dataSetName']
        self.data_csv_file = self.config['dataCsvFile']
        self.target_attributes = target_attributes

        self.attribute_configs = [ConfigAttribute(
            name=attr['name'],
            attr_type=attr['type'],
            data_type=attr['dataType'],
            order=attr.get("order", None),
            levels=attr.get("levels", None))
            for attr in self.config['attributeConfigs']]

    def get_attribute_config(self, name):
        for attr_config in self.attribute_configs:
            if attr_config.name == name:
                return attr_config
        return None

    def __get_attribute_config_by_type(self, type):
        retValue = []
        for attr_config in self.attribute_configs:
            if attr_config.type == type:
                retValue.append(attr_config)
        return retValue

    def __get_attribute_config_by_datatype(self, datatype):
        retValue = []
        for attr_config in self.attribute_configs:
            if attr_config.data_type == datatype:
                retValue.append(attr_config)
        return retValue

    def get_identifying_attributes(self):
        return self.__get_attribute_config_by_type('IDENTIFYING_ATTRIBUTE')

    def get_potentially_identifying_attributes(self):
        return self.__get_attribute_config_by_type('QUASI_IDENTIFYING_ATTRIBUTE')

    def get_categorical_attributes(self):
        return self.__get_attribute_config_by_datatype("categorical")

    def get_ordinal_attributes(self):
        return self.__get_attribute_config_by_datatype("ordinal")

    def get_date_attributes(self):
        return self.__get_attribute_config_by_datatype("date")

    def get_continuous_attributes(self):
        return self.__get_attribute_config_by_datatype("continuous")
