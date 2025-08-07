ds_info = {
    "Anderson2016": {
        "sep": "|",
        "skiprows": 1,
        "data_file": "Anderson2016/Data Tables/CGM.txt",
        "datetime_format": "%Y-%m-%d %H:%M:%S",
        "columns":{
            "ID": "DeidentID",
            "DateTime": "DisplayTime",
            "CGM": "CGM"
        }
    },
    "Brown2019": {
        "sep": "|",
        "skiprows": 1,
        "data_file": "Brown2019/Data Files/cgm.txt",
        "datetime_format": "%d%b%y:%H:%M:%S",
        "columns":{
            "ID": "PtID",
            "DateTime": "DataDtTm",
            "CGM": "CGM"
        }
    },
    "Buckinham2007": {
        "sep": ",",
        "skiprows": 1,
        "data_file": "Buckinham2007/DirecNetNavigatorPilotStudy/DataTables/tblFNavFreeStyle.csv",
        "datetime_format": ["%Y-%m-%d %H:%M:%S", "%H:%M:%S"],
        "columns":{
            "ID": "PtID",
            "DateTime": ["FreeReadDt", "FreeReadTm"],
            "CGM": "FreeResult"
        }
    },
    "Chase2005": {
        "sep": ",",
        "skiprows": 1,
        "data_file": "Chase2005/DirecNetOupatientRandomizedClinicalTrial/DataTables/tblCDataCGMS.csv",
        "datetime_format": ["%Y-%m-%d %H:%M:%S", "%I:%M %p"],
        "columns":{
            "ID": "PtID",
            "DateTime": ["ReadingDt", "ReadingTm"],
            "CGM": "SensorGLU"
        }
    },
    "Lynch2022": {
        "sep": "|",
        "skiprows": 1,
        "data_file": "Lynch2022/Data Tables/IOBP2DeviceCGM.txt",
        "datetime_format": "%m/%d/%Y %I:%M:%S %p",
        "columns":{
            "ID": "PtID",
            "DateTime": "DeviceDtTm",
            "CGM": "Value"
        }
    }
}

def get(dir):
    return ds_info[dir]