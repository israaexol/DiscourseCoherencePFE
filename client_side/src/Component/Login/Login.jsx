import React, { useState, useEffect, useCallback } from 'react'
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import FormControlLabel from "@mui/material/FormControlLabel";
import Checkbox from "@mui/material/Checkbox";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import LockIcon from '@mui/icons-material/Lock';
import Typography from "@mui/material/Typography";
import Slide from '@mui/material/Slide';
import { Alert, AlertTitle } from '@mui/material';
import { createTheme, ThemeProvider } from "@mui/material/styles";
import { alpha, styled } from '@mui/material/styles';
import axios from 'axios'

import './Login.css'

const EmailTextField = styled((props) => (
<TextField 
    variant="filled"
    margin="normal"
    required
    fullWidth
    id="username"
    label="Nom d'utilisateur"
    name="username"
    autoFocus
    InputProps={{ disableUnderline: true }} {...props} />
    ))(({ theme }) => ({
    '& .MuiFilledInput-root': {
    border: '2px solid #5885FB',
    overflow: 'hidden',
    borderRadius: 4,
    backgroundColor: 'white',
    transition: theme.transitions.create([
        'border-color',
        'background-color',
        'box-shadow',
    ]),
    '&:hover': {
        backgroundColor: 'white',
    },
    '&:focus': {
        boxShadow: '0 0 0 0.2rem rgba(0,123,255,.5)',
        backgroundColor: 'white',
    },
    },
}));

const PasswordTextField = styled((props) => (
    <TextField 
        variant="filled"
        margin="normal"
        required
        fullWidth
        name="password"
        label="Mot de passe"
        type="password"
        id="password"
        autoComplete="current-password"
        InputProps={{ disableUnderline: true }} {...props} />
        ))(({ theme }) => ({
        '& .MuiFilledInput-root': {
        border: '2px solid #5885FB',
        overflow: 'hidden',
        borderRadius: 4,
        backgroundColor: 'white',
        transition: theme.transitions.create([
            'border-color',
            'background-color',
            'box-shadow',
        ]),
        '&:hover': {
            backgroundColor: 'white',
        },
        '&:focus': {
            boxShadow: '0 0 0 0.2rem rgba(0,123,255,.5)',
            backgroundColor: 'white',
        },
        },
}));

const Login = (props) => {
  
    const [username, setUsername] = useState('')
    const [token, setToken] = useState('')
    const [password, setPassword] = useState('')

    const [errMsg, seterrMsg] = useState('')
    const [slide, setSlide] = useState(null)

    const onChangeHandler = event => {
        switch (event.target.name) {
            case "username":
                setUsername(event.target.value)
                console.log(username)
                break;
            case "password":
                setPassword(event.target.value)
                console.log(password)
                break;
            default:
                break;
        }
            
    }
    
    const Connexion = () => {
        var querystring = require('query-string');
        const bodyFormData = new URLSearchParams();
        bodyFormData.append('username', username)
        bodyFormData.append('password', password)
        axios.post(
            'http://localhost:8080/login', {
                data : bodyFormData
            },
            // querystring.stringify({
            //     "username": username,
            //     "password": password
            // }),
            { headers: { "Content-Type": "application/x-www-form-urlencoded" } },
        ).then(
            res => {
                alert("HELLO")
                console.log(res)
                if (res.success) {
                    setToken(res.access_token)
                    alert(token)
                    window.setTimeout( function(){
                        window.location.href = "/accueil";
                    }, 2000 );
                }
            }
        ).catch( err =>  {
                alert(err)
                console.log(err)
                setUsername("")
                setPassword("")
                seterrMsg("Le nom d'utilisateur ou le mot de passe n'est pas valide")
                setSlide(true)
            }
        )
    }   
  
    const message = (
        <div style={{margin:'10px 40px 30px 40px'}}>
            <Slide direction="up" in={slide} mountOnEnter unmountOnExit>
            <Alert severity="error">
                <strong>{errMsg}</strong>
            </Alert>
            </Slide>
        </div>
    )

    return (
        <>
        <div id="global">
            <Grid container component="main" className='root'
            spacing={0}
            direction="column"
            alignItems="center"
            justifyContent="center"
            mt={15}
            
            >
                <CssBaseline />
                <Grid
                className='size'
                item
                xs={12}
                sm={8}
                md={5}
                p={5}
                component={Paper}
                elevation={1}
                square
                direction="row"
                alignItems="center"
                justifyContent="center"
                display="flex"
                backgroundColor="#F8FAFF"
                >
                <div className='paper'>
                    <Avatar className='avatar' sx={{backgroundColor: "#5885FB"}}>
                        <LockIcon />
                    </Avatar>
                    <div>
                        <Typography component="h1" variant="h6" sx={{margin: "0 39%"}}>
                            Se connecter
                        </Typography>
                    </div>
                    <form className='form' noValidate>
                        <EmailTextField  onChange={onChangeHandler}/>
                        <PasswordTextField onChange={onChangeHandler}/>
                        <Button
                            type="submit"
                            fullWidth
                            variant="contained"
                            className='submit'
                            style={{
                                backgroundColor: "#5885FB",
                            }}
                            onClick = {Connexion}
                        >
                            Se connecter
                        </Button>
                        {message}
                        <h1>{token}</h1>
                    </form>
                </div>
                </Grid>
            </Grid>
        </div>
      </>
    );
    
}

export default Login


  