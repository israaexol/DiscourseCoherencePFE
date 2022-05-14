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

export const AjouterModele = ({handleCloseAjout}) => {

    const [state, setState] = useState({
        nom: '',
        description: '',
        exactitude: '',
        precision: '',
        rappel: '',
        scoref1: ''
    });

    const [errors, setErrors] = useState({})
    const [slide, setSlide] = useState(null)
    const [annuler, setAnnuler] = useState(null)
    const [modele, setModele] = useState(null)
    const [success, setSuccess] = useState(false)
    const { nom, description, exactitude, precision, rappel, scoref1 } = state;
    const values = { nom, description, exactitude, precision, rappel, scoref1 };

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
        if ('nom' in fieldValues)
            temp.nom = fieldValues.nom ? "" : "Ce champs est requis."
        if ('description' in fieldValues)
            temp.description = fieldValues.description ? "" : "Ce champs est requis."
        if ('exactitude' in fieldValues)
            temp.exactitude = fieldValues.exactitude ? "" : "Ce champs est requis."
        if ('precision' in fieldValues)
            temp.precision = fieldValues.precision ? "" : "Ce champs est requis."
        if ('rappel' in fieldValues)
            temp.rappel = fieldValues.rappel ? "" : "Ce champs est requis."
        if ('scoref1' in fieldValues)
            temp.scoref1 = fieldValues.scoref1 ? "" : "Ce champs est requis."
        setErrors({
            ...temp
        })

        if (fieldValues == values)
            return Object.values(temp).every(x => x == "")
    }
    
    const message = (
        <div style={{margin:'10px 40px 30px 40px'}}>
            <Slide direction="up" in={slide} mountOnEnter unmountOnExit>
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

    const addModele = useCallback(
        async () => {
            window.setTimeout( function(){ window.location.href = "/gestionmodeles" }, 1500 );
            // await axios.post(`${myServerBaseURL}/api/vehicules`, {
            //     nom: nom,
            //     description: description,
            //     exactitude: exactitude,
            //     precision: precision,
            //     rappel: rappel,
            //     scoref1: scoref1
            // },
        //     { headers : { authorization : `Basic ${getToken()}`}})
        //     .then((response) => {
        //         setSlide(true)
        //         setSuccess(true)
        //         console.log(response);
        //         window.setTimeout( function(){
        //             window.location.href = "/gestionmodeles";
        //         }, 2000 );
        //       }, (error) => {
        //         setSlide(true)
        //         setSuccess(false)
        //         console.log(error);
        //       });
        }
    )

    const onCreateNewModele = e => {
        e.preventDefault();
        if(validate()){
            addModele()
        } else {
            setSlide(true)
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
                                error={errors.nom === "" ? false : ""}
                                id="nom"
                                label="Nom du modèle"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('nom')}
                                value={values.nom}
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
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.exactitude === "" ? false : ""}
                                id="exactitude"
                                label="Exactitude"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('exactitude')}
                                defaultValue={values.exactitude}
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
                                error={errors.scoref1 === "" ? false : ""}
                                id="scoref1"
                                label="Score F1"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('scoref1')}
                                defaultValue={values.scoref1}
                            />
                        </div>
                        {message}
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