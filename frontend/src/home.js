import { useState, useEffect } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Paper, CardActionArea, CardMedia, Grid, TableContainer, Table, TableBody, TableHead, TableRow, TableCell, Button, CircularProgress, Box, Divider, Chip, Container } from "@material-ui/core";
import cblogo from "./cblogo.PNG";
import image from "./bg.png";
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
import Clear from '@material-ui/icons/Clear';
import EcoIcon from '@material-ui/icons/Eco';
import LocalHospitalIcon from '@material-ui/icons/LocalHospital';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
import SpeedIcon from '@material-ui/icons/Speed';
import DescriptionIcon from '@material-ui/icons/Description';
import BugReportIcon from '@material-ui/icons/BugReport';
import PhotoLibraryIcon from '@material-ui/icons/PhotoLibrary';

// Enhanced styled button with animation effects
const ColorButton = withStyles((theme) => ({
  root: {
    color: '#FFFFFF',
    backgroundColor: '#4CAF50',
    borderRadius: '8px',
    padding: '10px 20px',
    fontWeight: 'bold',
    boxShadow: '0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08)',
    transition: 'all 0.3s',
    '&:hover': {
      backgroundColor: '#3c8c40',
      transform: 'translateY(-2px)',
      boxShadow: '0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08)',
    },
    '&:active': {
      transform: 'translateY(1px)',
    },
  },
}))(Button);

const axios = require("axios").default;

const useStyles = makeStyles((theme) => ({
  grow: {
    flexGrow: 1,
  },
  clearButton: {
    width: "-webkit-fill-available",
    borderRadius: "10px",
    padding: "12px 20px",
    color: "#FFFFFF",
    fontSize: "16px",
    fontWeight: 700,
    textTransform: 'none',
    marginTop: '20px',
    boxShadow: '0 4px 14px 0 rgba(76, 175, 80, 0.39)',
  },
  root: {
    maxWidth: 345,
    flexGrow: 1,
  },
  media: {
    height: '100%',
    minHeight: 400,
    borderRadius: '20px',
    boxShadow: 'inset 0 -10px 20px rgba(0,0,0,0.12)',
    objectFit: 'contain',
  },
  mainContainer: {
    backgroundImage: `url(${image})`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    height: "100vh",
    width: "100%",
    position: "fixed",
    top: 0,
    left: 0,
    zIndex: -1,
  },
  content: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "flex-start",
    padding: "80px 20px 20px 20px",
    overflow: "auto",
    overflowX: "auto",
    width: "100%",
  },
  mainCard: {
    width: "100%",
    maxWidth: "80%",
    maxHeight: "90vh",
    background: 'rgba(255, 255, 255, 0.85)',
    backdropFilter: 'blur(10px)',
    boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
    borderRadius: '20px',
    border: '1px solid rgba(255, 255, 255, 0.18)',
    overflow: 'auto',
    transition: 'all 0.3s ease',
    display: 'flex',
    flexDirection: 'column',
    padding: '30px',
    margin: '20px 0',
  },
  sectionTitle: {
    fontSize: '20px',
    fontWeight: 700,
    marginBottom: '20px',
    textAlign: 'center',
    color: '#4CAF50',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    '& svg': {
      marginRight: '8px',
      fontSize: '24px',
    },
  },
  instructionsContainer: {
    marginBottom: '20px',
    width: '100%',
  },
  stepsContainer: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    flexWrap: 'wrap',
    gap: '15px',
    marginBottom: '20px',
  },
  stepBox: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    backgroundColor: 'rgba(76, 175, 80, 0.08)',
    padding: '15px',
    borderRadius: '12px',
    border: '1px solid rgba(76, 175, 80, 0.1)',
    width: '30%',
    minWidth: '160px',
    transition: 'all 0.3s ease',
    '&:hover': {
      transform: 'translateY(-5px)',
      boxShadow: '0 5px 15px rgba(76, 175, 80, 0.2)',
      backgroundColor: 'rgba(76, 175, 80, 0.12)',
    },
  },
  stepNumber: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: '16px',
    backgroundColor: '#4CAF50',
    borderRadius: '50%',
    width: '28px',
    height: '28px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '10px',
  },
  stepTitle: {
    color: '#4CAF50',
    fontWeight: 'bold',
    marginBottom: '5px',
    textAlign: 'center',
  },
  stepDescription: {
    fontSize: '12px',
    textAlign: 'center',
    color: '#666',
  },
  uploadSection: {
    width: '100%',
    margin: '0 auto 30px auto',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    [theme.breakpoints.down('xs')]: {
      padding: '0 10px',
    },
    [theme.breakpoints.between('sm', 'md')]: {
      width: '90%',
    },
    [theme.breakpoints.up('md')]: {
      width: '80%',
    },
  },
  uploadCard: {
    padding: '25px',
    borderRadius: '15px',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
    border: '1px solid rgba(76, 175, 80, 0.3)',
    width: '100%',
    maxWidth: '550px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    [theme.breakpoints.down('xs')]: {
      padding: '20px 15px',
    },
  },
  dropZone: {
    width: '70%',
    minHeight: '180px',
    border: '2px dashed #4CAF50',
    borderRadius: '12px',
    backgroundColor: 'rgba(76, 175, 80, 0.04)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    margin: '10px 0',
    transition: 'all 0.3s ease',
    [theme.breakpoints.down('xs')]: {
      minHeight: '150px',
      padding: '15px',
    },
    [theme.breakpoints.up('md')]: {
      minHeight: '200px',
    },
    '&:hover': {
      backgroundColor: 'rgba(76, 175, 80, 0.08)',
      borderColor: '#4CAF50',
      transform: 'translateY(-2px)',
      boxShadow: '0 5px 15px rgba(76, 175, 80, 0.1)',
    },
  },
  iconContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    margin: '10px 0 20px 0',
    width: '60px',
    height: '60px',
    borderRadius: '50%',
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
  },
  uploadIcon: {
    fontSize: '35px',
    color: '#4CAF50',
    animation: '$pulse 2s infinite',
    [theme.breakpoints.down('xs')]: {
      fontSize: '28px',
    },
  },
  uploadTitle: {
    fontSize: '22px',
    fontWeight: 600,
    color: '#4CAF50',
    marginBottom: '12px',
    textAlign: 'center',
    [theme.breakpoints.down('xs')]: {
      fontSize: '18px',
      marginBottom: '8px',
    },
  },
  uploadSubtitle: {
    fontSize: '15px',
    color: '#666',
    marginBottom: '20px',
    textAlign: 'center',
    maxWidth: '90%',
    lineHeight: '1.4',
    [theme.breakpoints.down('xs')]: {
      fontSize: '13px',
      marginBottom: '15px',
    },
  },
  '@keyframes pulse': {
    '0%': {
      transform: 'scale(1)',
      opacity: 1
    },
    '50%': {
      transform: 'scale(1.1)',
      opacity: 0.8
    },
    '100%': {
      transform: 'scale(1)',
      opacity: 1
    }
  },
  imagePreviewContainer: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  imagePreview: {
    width: '100%',
    maxWidth: '400px',
    height: '300px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    backgroundSize: 'contain',
    backgroundPosition: 'center',
    backgroundRepeat: 'no-repeat',
    borderRadius: '15px',
    border: '1px solid rgba(76, 175, 80, 0.3)',
    margin: '0 auto 15px auto',
  },
  resultSection: {
    width: '100%',
    marginTop: '20px',
  },
  loaderContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px 0',
  },
  loader: {
    color: '#4CAF50 !important',
    margin: '20px 0',
  },
  infoCard: {
    backgroundColor: 'rgba(76, 175, 80, 0.05)',
    borderRadius: '12px',
    padding: '12px',
    marginBottom: '15px',
    border: '1px solid rgba(76, 175, 80, 0.1)',
    width: '100%',
  },
  infoHeader: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '10px',
    color: '#4CAF50',
    fontWeight: 600,
    '& svg': {
      marginRight: '8px',
    },
  },
  tableContainer: {
    backgroundColor: 'transparent !important',
    boxShadow: 'none !important',
    borderRadius: '10px',
    overflow: 'hidden',
    width: '100%',
    margin: '10px 0',
  },
  table: {
    backgroundColor: 'transparent !important',
  },
  tableRow: {
    backgroundColor: 'transparent !important',
    transition: 'background 0.2s',
    '&:hover': {
      backgroundColor: 'rgba(76, 175, 80, 0.04) !important',
    },
  },
  tableCell: {
    fontSize: '14px',
    backgroundColor: 'transparent !important',
    borderColor: 'rgba(224, 224, 224, 0.3) !important',
    color: '#333 !important',
    fontWeight: 600,
    padding: '10px 16px',
  },
  tableCell1: {
    fontSize: '13px',
    backgroundColor: 'transparent !important',
    borderColor: 'rgba(224, 224, 224, 0.3) !important',
    color: '#666 !important',
    fontWeight: 600,
    padding: '10px 16px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  confidenceChip: {
    fontWeight: 'bold',
    backgroundColor: '#4CAF50',
    color: 'white',
    padding: '4px 12px',
    borderRadius: '16px',
    display: 'inline-block',
    fontSize: '14px',
  },
  appbar: {
    background: 'linear-gradient(45deg, #4CAF50 30%, #81C784 90%)',
    boxShadow: '0 3px 5px 2px rgba(76, 175, 80, .3)',
    color: 'white',
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    zIndex: 100,
  },
  title: {
    fontWeight: 600,
    letterSpacing: '0.5px',
    display: 'flex',
    alignItems: 'center',
  },
  titleIcon: {
    marginRight: '10px',
    fontSize: '28px',
  },
}));

export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
  const [isLoading, setIsloading] = useState(false);
  let confidence = 0;

  const sendFile = async () => {
    if (image) {
      let formData = new FormData();
      formData.append("file", selectedFile);
      let res = await axios({
        method: "post",
        url: process.env.REACT_APP_API_URL,
        data: formData,
      });
      if (res.status === 200) {
        setData(res.data);
      }
      setIsloading(false);
    }
  }

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) {
      return;
    }
    setIsloading(true);
    sendFile();
  }, [preview]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setImage(true);
  };

  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  return (
    <React.Fragment>
      <div className={classes.mainContainer} />
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography className={classes.title} variant="h6" noWrap>
            <EcoIcon className={classes.titleIcon} />
            Plant Disease Detector
          </Typography>
          <div className={classes.grow} />
          <Avatar src={cblogo} alt="Logo" />
        </Toolbar>
      </AppBar>
      <div className={classes.content}>
        <Card className={classes.mainCard}>
          {/* Instructions Section */}
          <div className={classes.instructionsContainer}>
            <Typography className={classes.sectionTitle}>
              <PhotoLibraryIcon />
              How It Works
            </Typography>
            <div className={classes.stepsContainer}>
              <div className={classes.stepBox}>
                <span className={classes.stepNumber}>1</span>
                <Typography variant="subtitle2" className={classes.stepTitle}>Upload Image</Typography>
                <Typography variant="body2" className={classes.stepDescription}>
                  Take a clear photo of the plant leaf and upload it to our system
                </Typography>
              </div>
              <div className={classes.stepBox}>
                <span className={classes.stepNumber}>2</span>
                <Typography variant="subtitle2" className={classes.stepTitle}>AI Analysis</Typography>
                <Typography variant="body2" className={classes.stepDescription}>
                  Our AI model analyzes the leaf image to identify diseases
                </Typography>
              </div>
              <div className={classes.stepBox}>
                <span className={classes.stepNumber}>3</span>
                <Typography variant="subtitle2" className={classes.stepTitle}>View Results</Typography>
                <Typography variant="body2" className={classes.stepDescription}>
                  Get detailed diagnosis with treatment recommendations
                </Typography>
              </div>
            </div>
          </div>

          {/* Upload Section */}
          {!image ? (
            <div className={classes.uploadSection}>
              <Paper elevation={0} className={classes.uploadCard}>
                <Typography variant="h6" className={classes.uploadTitle}>
                  Identify Plant Disease
                </Typography>
                <Typography variant="body2" className={classes.uploadSubtitle}>
                  Upload a clear image of the affected leaf for instant AI diagnosis
                </Typography>
                {/* <div className={classes.iconContainer}>
                    <CloudUploadIcon className={classes.uploadIcon} />
                  </div> */}
                <div className={classes.dropZone} style={{flexDirection: 'row', alignItems: 'center', justifyContent: 'center'}}>
                  <DropzoneArea style={{width: '100%', height: '100%', marginTop: '100px'}}
                    acceptedFiles={['image/*']}
                    dropzoneText={"Drag & drop leaf image here"}
                    onChange={onSelectFile}
                    filesLimit={1}
                    showPreviewsInDropzone={false}
                    showAlerts={false}
                    maxFileSize={5000000}
                    useChipsForPreview={true}
                    previewGridProps={{ container: { spacing: 1, direction: 'row' } }}
                  />
                 
                </div>
              </Paper>
            </div>
          ) : (
            <div className={classes.imagePreviewContainer}>
              {isLoading ? (
                <div className={classes.loaderContainer}>
                  <CircularProgress size={60} className={classes.loader} />
                  <Typography variant="h6" style={{ marginTop: '20px', color: '#4CAF50' }}>
                    Analyzing Image
                  </Typography>
                  <Typography variant="body2" style={{ color: '#666' }}>
                    Please wait while our AI processes your plant image
                  </Typography>
                </div>
              ) : (
                <>
                  <Typography variant="h6" gutterBottom style={{ color: '#4CAF50', fontWeight: 600, textAlign: 'center', marginBottom: '15px' }}>
                    Plant Leaf Analysis
                  </Typography>
                  <div
                    className={classes.imagePreview}
                    style={{ backgroundImage: `url(${preview})` }}
                  />
                  <ColorButton
                    variant="contained"
                    onClick={clearData}
                    startIcon={<Clear />}
                    style={{ marginTop: '10px' }}
                  >
                    Clear and Upload New Image
                  </ColorButton>
                </>
              )}
            </div>
          )}

          {/* Results Section */}
          {data && !isLoading && (
            <div className={classes.resultSection}>
              <Typography className={classes.sectionTitle}>
                <BugReportIcon />
                Disease Diagnosis
              </Typography>

              <div className={classes.infoCard}>
                <div className={classes.infoHeader}>
                  <DescriptionIcon fontSize="small" />
                  <Typography variant="subtitle2">Disease Information</Typography>
                </div>
                <TableContainer component={Paper} className={classes.tableContainer}>
                  <Table className={classes.table} size="small">
                    <TableBody>
                      <TableRow className={classes.tableRow}>
                        <TableCell className={classes.tableCell1}>Disease</TableCell>
                        <TableCell className={classes.tableCell}>
                          {data.disease || data.class}
                        </TableCell>
                      </TableRow>
                      <TableRow className={classes.tableRow}>
                        <TableCell className={classes.tableCell1}>Confidence</TableCell>
                        <TableCell className={classes.tableCell}>
                          <span className={classes.confidenceChip}>{confidence}%</span>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </div>

              {data.medicine && (
                <div className={classes.infoCard}>
                  <div className={classes.infoHeader}>
                    <LocalHospitalIcon fontSize="small" />
                    <Typography variant="subtitle2">Treatment</Typography>
                  </div>
                  <TableContainer component={Paper} className={classes.tableContainer}>
                    <Table className={classes.table} size="small">
                      <TableBody>
                        <TableRow className={classes.tableRow}>
                          <TableCell className={classes.tableCell1}>Medicine</TableCell>
                          <TableCell className={classes.tableCell}>{data.medicine.name}</TableCell>
                        </TableRow>
                        {data.medicine.brand && (
                          <TableRow className={classes.tableRow}>
                            <TableCell className={classes.tableCell1}>Brand</TableCell>
                            <TableCell className={classes.tableCell}>{data.medicine.brand}</TableCell>
                          </TableRow>
                        )}
                        {data.medicine.description && (
                          <TableRow className={classes.tableRow}>
                            <TableCell className={classes.tableCell1}>Description</TableCell>
                            <TableCell className={classes.tableCell} style={{ fontSize: '13px' }}>{data.medicine.description}</TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </div>
              )}
            </div>
          )}

          {/* Empty State for Result Section */}
          {!data && !isLoading && image && (
            <Box className={classes.resultSection} textAlign="center">
              <Typography variant="body1" color="textSecondary">
                Waiting for analysis results...
              </Typography>
            </Box>
          )}
        </Card>
      </div>
    </React.Fragment>
  );
};

