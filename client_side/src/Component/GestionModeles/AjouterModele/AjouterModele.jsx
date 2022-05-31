import React, { useState, useCallback, useEffect, useRef } from 'react'
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import MenuItem from '@mui/material/MenuItem';
import Slide from '@mui/material/Slide';
import { Alert, AlertTitle } from '@mui/material';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogTitle from '@mui/material/DialogTitle';
import { Typography } from '@mui/material';
import axios from 'axios'
import Link from '@mui/material/Link';
import { styled } from '@mui/material/styles';


export const AjouterModele = ({handleCloseAjout}) => {

    const [state, setState] = useState({
        id: 55,
        name: '',
        preprocess : '',
        description: '',
        accuracy: '',
        precision: '',
        rappel: '',
        F1_score: '',
        visibility: true
    });

    const hiddenFileInput = React.useRef(null); 
    const [errors, setErrors] = useState({})
    const [slide, setSlide] = useState(null)
    const [slideErr, setSlideErr] = useState(null)
    const [annuler, setAnnuler] = useState(null)
    const [modele, setModele] = useState(null)
    const [success, setSuccess] = useState(false)
    const [data, setData] = useState(null)
    const [fileName, setFileName] = useState(null)
    const { name, description, accuracy, precision, rappel, F1_score, preprocess } = state;
    const values = { name, description, accuracy, precision, rappel, F1_score, preprocess };

    // handle fields change
    const handleChange = input => e => {
        setState({...state, [input]: e.target.value})
    }

    const handleOpenAnnuler = () => {
        setAnnuler(true)
    }

    const handleCloseAnnuler = () => {
        setAnnuler(false)
    }

    const validate = (fieldValues = values) => {
        let temp = { ...errors }
        if ('name' in fieldValues)
            temp.name = fieldValues.name ? "" : "Ce champs est requis."
        if ('description' in fieldValues)
            temp.description = fieldValues.description ? "" : "Ce champs est requis."
        if ('accuracy' in fieldValues)
            temp.accuracy = fieldValues.accuracy ? "" : "Ce champs est requis."
        if ('precision' in fieldValues)
            temp.precision = fieldValues.precision ? "" : "Ce champs est requis."
        if ('rappel' in fieldValues)
            temp.rappel = fieldValues.rappel ? "" : "Ce champs est requis."
        if ('F1_score' in fieldValues)
            temp.F1_score = fieldValues.F1_score ? "" : "Ce champs est requis."
        if ('preprocess' in fieldValues)
            temp.preprocess = fieldValues.preprocess ? "" : "Ce champs est requis."
        if (!fileName)
            temp.file_name = "Ce champs est requis."
            // "hybridation": hybridation,
        setErrors({
            ...temp
        })

        if (fieldValues == values)
            return Object.values(temp).every(x => x == "")
    }
    
    const message = (
        <div style={{margin:'10px 40px 30px 40px'}}>
            <Slide direction="up" in={slideErr} mountOnEnter unmountOnExit>
                <Alert severity="error">
                    <strong>Veuillez renseigner les champs requis.</strong>
                </Alert>
            </Slide>
        </div>

    )

    const successMessage = (
        <div style={{margin:'10px 40px 30px 40px'}}>
            {success && (
                <Slide direction="up" in={slide} mountOnEnter unmountOnExit>
                    <Alert severity="success">
                        <AlertTitle>Succés</AlertTitle>
                        Le modèle a été ajouté <strong>avec succés</strong>
                    </Alert>
                </Slide>
                ) } { !success && (
                <Slide direction="up" in={slide} mountOnEnter unmountOnExit>
                    <Alert severity="error">
                        <AlertTitle>Erreur!</AlertTitle>
                        <strong>Le modèle n'a pas été ajouté avec succés</strong>
                    </Alert>
                </Slide>
            ) }
        </div>

    )

    const annulerDialogue = (
        <div>
            <Dialog
                open={annuler}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <Typography style={{fontFamily:'Poppins', fontSize:'15px', padding:'14px 20px', boxShadow:'none'}}>
                    Voulez-vous vraiment annuler l'ajout d'un nouveau modèle? 
                    <br></br>
                    Toutes les informations saisies seront perdues.
                </Typography>                    
                <DialogActions>
                    <Button onClick={handleCloseAjout} style={{textTransform:"capitalize", color:"#F5365C", fontFamily:'Poppins', margin:"12px 20px", fontWeight:"bold"}}>
                        Oui
                    </Button>
                    <Button onClick={handleCloseAnnuler} style={{textTransform:"capitalize", backgroundColor:"#252834", color:"white", fontFamily:'Poppins', padding:"6px 12px", margin:"12px 20px"}}>
                        Non
                    </Button>
                </DialogActions>
            </Dialog>
        </div>
    )

    const Input = styled('input')({
        display: 'none',
      });

    const handleImport = event => {
        hiddenFileInput.current.click();
        

    };

    const handleFileChange = event => {
        const fileUploaded = event.target.files[0];
        if (fileUploaded) {
            let dataFile = new FormData();
            console.log(fileUploaded)
            dataFile.append('pickle', fileUploaded);
            // while(Object.keys(dataFile).length === 0) {

                
            //     console.log(dataFile)
            // }
            
            setData(dataFile)
            setFileName(fileUploaded.name)
            console.log(fileName)
        }
    };

    const uploadFile = () => {
        axios
        .post('http://localhost:8080/addpickle_model', data)
        .then((res) => {
          console.log("fichier " + fileName +" importé")
        })
        .catch((error) => {
          alert(`Error: ${error.message}`)
        })
    }

    const addModele = useCallback(
        async () => {
            uploadFile()
            await axios.post('http://localhost:8080/add_model', {
                "id": 55,
                "name" : name,
                "description" : description,
                "F1_score": F1_score,
                "precision" : precision,
                "accuracy" : accuracy,
                "rappel": rappel,
                "file_name": fileName,
                "preprocess" : preprocess,
                "hybridation": false,
                "visibility": true
            })
            .then((response) => {
                setSlide(true)
                setSuccess(true)
                console.log(response);
                window.setTimeout( function(){
                    window.location.href = "/gestionmodeles";
                }, 2000 );
              }, (error) => {
                setSlideErr(true)
                setSuccess(false)
                console.log(error);
              });
        }
    )

    const onCreateNewModele = e => {
        e.preventDefault();
        if(validate()){
            addModele()
        } else {
            setSlideErr(true)
        }
    }

    return (
             <Container fluid style={{paddingBottom:"40px"}}>
                    <div style={{padding:"10px", fontFamily: 'Poppins'}}>
                        <h4 style={{margin:"10px 30% 0px"}}>Nouveau modèle</h4>
                    </div>
                    <form noValidate="false">
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.name === "" ? false : ""}
                                id="name"
                                label="Nom du modèle"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('name')}
                                value={values.name}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.preprocess === "" ? false : ""}
                                id="preprocess"
                                label="Niveau de prétraitement"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('preprocess')}
                                value={values.preprocess}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.description === "" ? false : ""}
                                id="description"
                                label="Description"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('description')}
                                defaultValue={values.description}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.accuracy === "" ? false : ""}
                                id="accuracy"
                                label="Exactitude"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('accuracy')}
                                defaultValue={values.accuracy}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.precision === "" ? false : ""}
                                id="precision"
                                label="Précision"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('precision')}
                                defaultValue={values.precision}
                                type='string'
                            />
                        </div>
                        <br></br> 
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.rappel === "" ? false : ""}
                                id="rappel"
                                label="Rappel"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('rappel')}
                                defaultValue={values.rappel}
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.F1_score === "" ? false : ""}
                                id="F1_score"
                                label="Score F1"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('F1_score')}
                                defaultValue={values.F1_score}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div className="flex-container" style={{display: "flex", flexWrap:'wrap', gap:'30px', justifyContent:'center', alignItems:'center'}}>
                            <div>
                                <Input ref={hiddenFileInput} onChange={handleFileChange} id="pickle-file" type="file" />
                                <Button onClick={handleImport} style={{backgroundColor:"#007bff", textTransform:"capitalize", color:"white", fontWeight:'bold'}} variant="contained">
                                {fileName ?? "Importer un modèle"}
                                </Button>
                            </div>
                        </div>
                        {message}
                        {successMessage}
                        <div className="flex-container" style={{display: "flex", flexWrap:'wrap', gap:'30px', justifyContent:'center', alignItems:'center'}}>
                            <div>
                                <Button onClick={handleOpenAnnuler} style={{backgroundColor:"#F5365C", textTransform:"capitalize", color:"white", fontWeight:'bold'}} variant="contained">
                                    Annuler
                                </Button>
                            </div>
                        <div>
                        <Button onClick={onCreateNewModele} style={{backgroundColor:"#00B668", textTransform:"capitalize", color:"white", fontWeight:'bold', width:'150px'}} variant="contained">
                            <Link to="/gestionmodeles" variant="contained" style={{fontFamily:'Poppins', color:'white'}}>
                                Confirmer
                            </Link>
                            </Button>
                        </div>
                        </div>
                        {annulerDialogue}
                    </form>
            </Container>
    )
}

export default AjouterModele