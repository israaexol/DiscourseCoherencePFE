import React, { useState, useEffect, useCallback } from 'react'
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import Slide from '@mui/material/Slide';
import { Alert, AlertTitle } from '@mui/material';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import { Typography } from '@mui/material';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import axios from "axios";

const ModifierModele = (props) => {

  const [state, setState] = useState({});
  const [modelID, setModelID] = useState('')

  const getModele = () => {
        if(props.modele) {
            setState(props.modele)
            setModelID(props.modele.id)
        }
        console.log(state)
    }
  useEffect(() => {
    getModele();
  }, []);

  // handle fields change
  const handleChange = input => e => {
      setState({...state, [input]: e.target.value})
      console.log(state)
  }

  const handleCloseModif = props.handleCloseModif

  const [errors, setErrors] = useState({})
  const [slide, setSlide] = useState(null)
  const [annuler, setAnnuler] = useState(null)
  const [confirmer, setConfirmer] = useState(null)

  const handleOpenAnnuler = () => {
      setAnnuler(true)
  }

  const handleCloseAnnuler = () => {
      setAnnuler(false)
  }

  const handleOpenConfirmer = () => {
      setConfirmer(true)
  }

  const handleCloseConfirmer = () => {
      setConfirmer(false)
  }

  const validate = (fieldValues = state) => {
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
    if ('visibility' in fieldValues)
          temp.visibility = fieldValues.visibility ? "" : "Ce champs est requis."
      setErrors({
          ...temp
      })

      if (fieldValues == state)
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

  const [slideModif, setSlideModif] = useState(null)
  const [modifSuccess, setModifSuccess] = useState(null)
  const modifSuccessMessage = (
      <div style={{margin:'20px 0px', padding:'12px'}}>
                {(modifSuccess == true) && (
                  <Slide direction="up" in={slideModif} mountOnEnter unmountOnExit>
                  <Alert severity="success" onClose={() => {
                      setSlideModif(false)
                      }}>
                      <AlertTitle>Succés</AlertTitle>
                      Le modèle a été modifié <strong>avec succés</strong>
                  </Alert>
                  </Slide>
                ) } { (modifSuccess == false) && (
                  <Slide direction="up" in={slideModif} mountOnEnter unmountOnExit>
                  <Alert severity="error">
                      <AlertTitle>Erreur!</AlertTitle>
                      <strong>Erreur lors de la modification du modèle</strong>
                  </Alert>
                  </Slide>
                ) }
      </div>
    )
    
    const onModifierModele = useCallback( 
      async () => {
        window.setTimeout( function(){
                            handleCloseConfirmer()
                            setModifSuccess(null)
                            window.location.href = "/gestionmodeles";
                          }, 2000 );
        const response = await axios.put(`http://localhost:8080/update_model/${modelID}`, {
            "id": state.id,
            "name": state.name,
            "description": state.description,
            "F1_score": state.F1_score,
            "precision": state.precision,
            "rappel": state.rappel,
            "accuracy": state.accuracy,
            "visibility" : state.visibility
            
            
          })
            .then((response) => {
                setSlideModif(true)
                setModifSuccess(true)
                console.log("modifié")
                console.log(response);
                window.setTimeout( function(){
                    handleCloseConfirmer()
                    setModifSuccess(null)
                    window.location.href = "/gestionmodeles";
                }, 2000 );
                }, (error) => {
                setSlideModif(true)
                setModifSuccess(false)
                console.log("erreur")
                console.log(error);
                window.setTimeout( function(){
                    handleCloseConfirmer()
                    setModifSuccess(null)
                }, 2000 );
                });
      });

    const continuer = e => {
        e.preventDefault();
        if(validate()){
            setSlide(null)
            handleOpenConfirmer()
        } else {
            setSlide(true)
        }
    }

    const annulerDialogue = (
        <div>
            <Dialog
                open={annuler}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
            <Typography style={{fontFamily:'Poppins', fontSize:'18px', padding:'14px 20px', boxShadow:'none'}}>
                    Voulez-vous vraiment annuler la modification du modèle? 
                    <br></br>
                    Toutes les informations saisies seront perdues.
            </Typography>                    
                <DialogActions>
                <Button onClick={handleCloseModif} style={{textTransform:"capitalize", color:"#F5365C", fontFamily:'Poppins', margin:"12px 20px", fontWeight:"bold"}}>
                    Oui
                </Button>
                <Button onClick={handleCloseAnnuler} style={{textTransform:"capitalize", backgroundColor:"#252834", color:"white", fontFamily:'Poppins', padding:"6px 12px", margin:"12px 20px"}}>
                    Non
                </Button>
                </DialogActions>
            </Dialog>
        </div>
    )

    const confirmerDialogue = (
        <div>
            <Dialog
                open={confirmer}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
            <Typography style={{fontFamily:'Poppins', fontSize:'17px', padding:'20px 17px 18px', boxShadow:'none'}}>
                    Voulez-vous confirmer la modification du modèle? 
                </Typography>                    
                <DialogActions>
                <Button onClick={onModifierModele} style={{textTransform:"capitalize", color:"#F5365C", fontFamily:'Poppins', margin:"12px 12px", fontWeight:"bold"}}>
                    Oui
                </Button>
                <Button onClick={handleCloseConfirmer} style={{textTransform:"capitalize", backgroundColor:"#252834", color:"white", fontFamily:'Poppins', padding:"6px 12px", margin:"12px 20px"}}>
                    Non
                </Button>
                </DialogActions>
                {modifSuccess ? modifSuccessMessage : <br></br>}
            </Dialog>
        </div>
    )
    
        return (
             <Container fluid style={{paddingBottom:"40px"}}>
                    <div style={{padding:"10px", fontFamily:"Poppins"}}>
                    <h5 style={{margin:"10px 15% 9px"}}>Modifier les informations du modèle</h5>
                    </div>
                    <form noValidate="false">
                        <div style={{padding:"5px 40px"}}>
                          {/* <InputLabel>Nom du modèle</InputLabel> */}
                          <TextField
                              required
                              error={errors.name === "" ? false : ""}
                              id="name"
                              variant="outlined"
                              label="Nom du modèle"
                              InputLabelProps={{
                                shrink: true,
                              }}
                              fullWidth='true'
                              value={state.name}
                              onChange={handleChange('name')}
                          />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                              required
                              error={errors.description === "" ? false : ""}
                              id="description"
                              variant="outlined"
                              label="Description"
                              InputLabelProps={{
                                shrink: true,
                              }}
                              fullWidth='true'
                              value={state.description}
                              onChange={handleChange('description')}
                          />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <InputLabel id="demo-simple-select-label">Visibilité</InputLabel>
                            <Select
                                required
                                error={errors.visibility === "" ? false : ""}
                                id="visibility"
                                variant="outlined"
                                value={state.visibility}
                                label="Visibilité"
                                onChange={handleChange('visibility')}
                                fullWidth='true'
                            >
                            <MenuItem value={true}>Oui</MenuItem>
                            <MenuItem value={false}>Non</MenuItem>
                            </Select>
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                                required
                                error={errors.accuracy === "" ? false : ""}
                                id="accuracy"
                                variant="outlined"
                                label="Exactitude"
                                InputLabelProps={{
                                  shrink: true,
                                }}
                                fullWidth='true'
                                value={state.accuracy}
                                onChange={handleChange('accuracy')}
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                                required
                                error={errors.precision === "" ? false : ""}
                                id="precision"
                                variant="outlined"
                                label="Précision"
                                InputLabelProps={{
                                  shrink: true,
                                }}
                                fullWidth='true'
                                value={state.precision}
                                onChange={handleChange('precision')}
                              />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                              required
                              error={errors.rappel === "" ? false : ""}
                              id="rappel"
                              variant="outlined"
                              label="Rappel"
                              InputLabelProps={{
                                shrink: true,
                              }}
                              fullWidth='true'
                              onChange={handleChange('rappel')}
                              value={state.rappel}
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                                required
                                error={errors.F1_score === "" ? false : ""}
                                id="F1_score"
                                variant="outlined"
                                label="Score F1"
                                InputLabelProps={{
                                  shrink: true,
                                }}
                                fullWidth='true'
                                value={state.F1_score}
                                onChange={handleChange('F1_score')}
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
                                <Button onClick={continuer} style={{backgroundColor:"#00B668", textTransform:"capitalize", color:"white", fontWeight:'bold', width:'150px'}} variant="contained">
                                    Sauvegarder
                                </Button>
                                </div>
                        </div>
                        {annulerDialogue}
                        {confirmerDialogue}

                    </form>
            </Container>
        )
  }

export default ModifierModele