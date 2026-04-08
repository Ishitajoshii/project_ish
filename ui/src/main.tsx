import {
  StrictMode as strict_mode,
  createElement as create_element,
} from 'react'
import { createRoot as create_root } from 'react-dom/client'
import './index.css'
import { app } from './circuit_rl_app'

create_root(document.getElementById('root')!).render(
  create_element(strict_mode, null, create_element(app)),
)
